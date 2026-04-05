from pathlib import Path
import csv
import shutil
import cv2
import numpy as np
import logging
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

from pipelines.base import Stage


# LOGGING SET UP
LOG_PATH = Path("pipeline.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# AUTO LABEL STAGE
class AutoLabelStage(Stage):
    def __init__(self, raw_dir: Path, output_csv: Path, confidence_threshold=0.7):
        super().__init__("AutoLabel")
        self.raw_dir = Path(raw_dir)
        self.output_csv = Path(output_csv)
        self.confidence_threshold = confidence_threshold
        self.model = MobileNetV2(weights="imagenet")

    def requires(self):
        return [self.raw_dir]

    def produces(self):
        return [self.output_csv]

    def is_complete(self):
        return self.output_csv.exists()

    def _load_and_preprocess(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        return arr

    def _map_to_binary(self, decoded_pred):
        class_name = decoded_pred[0][1].lower()
        confidence = decoded_pred[0][2]

        if "cat" in class_name:
            return "cat", confidence
        else:
            return "not_cat", confidence

    def run(self):
        logger.info("AutoLabelStage started.")
        image_paths = list(self.raw_dir.glob("*"))

        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "confidence"])

            for image_path in tqdm(image_paths, desc="AutoLabel"):
                try:
                    arr = self._load_and_preprocess(image_path)
                    preds = self.model.predict(arr, verbose=0)
                    decoded = decode_predictions(preds, top=1)[0]
                    label, confidence = self._map_to_binary(decoded)

                    if confidence >= self.confidence_threshold:
                        writer.writerow([image_path.name, label, confidence])

                except Exception as e:
                    logger.warning(f"AutoLabel failed for {image_path.name}: {e}")
                    continue

        logger.info("AutoLabelStage finished.")

    def validate(self):
        if not self.output_csv.exists():
            raise RuntimeError("Auto-label CSV not created.")

        with open(self.output_csv, "r") as f:
            rows = list(csv.reader(f))

        if len(rows) <= 1:
            raise RuntimeError("Auto-label CSV is empty.")

        logger.info(f"AutoLabelStage validation passed: {len(rows)-1} labels written.")


# VERIFY STAGE
class VerifyStage(Stage):
    def __init__(self, raw_dir: Path, auto_csv: Path, verified_output_csv: Path):
        super().__init__("Verify")
        self.raw_dir = Path(raw_dir)
        self.auto_csv = Path(auto_csv)
        self.verified_output_csv = Path(verified_output_csv)

    def requires(self):
        return [self.raw_dir, self.auto_csv]

    def produces(self):
        return [self.verified_output_csv]

    def is_complete(self):
        return False  # manual stage, never auto-skip

    def run(self):
        logger.info("VerifyStage started.")

        all_rows = self._load_auto_labels()
        verified_set = set()

        if self.verified_output_csv.exists():
            with open(self.verified_output_csv, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    verified_set.add(row[0])

        mode = "a" if self.verified_output_csv.exists() else "w"

        with open(self.verified_output_csv, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(["filename", "label"])

            for filename, predicted_label, _ in tqdm(all_rows, desc="Verify"):
                if filename in verified_set:
                    continue

                image_path = self.raw_dir / filename
                if not image_path.exists():
                    continue

                verified_label = self._verify_one(image_path, predicted_label)

                if verified_label is None:
                    logger.info("VerifyStage interrupted by user.")
                    return

                if verified_label == "skip":
                    continue

                writer.writerow([filename, verified_label])
                f.flush()

        logger.info("VerifyStage finished.")

    def validate(self):
        if not self.verified_output_csv.exists():
            raise RuntimeError("Verified CSV not created.")

        with open(self.verified_output_csv, "r") as f:
            rows = list(csv.reader(f))

        if len(rows) <= 1:
            raise RuntimeError("Verified CSV is empty.")

        logger.info(f"VerifyStage validation passed: {len(rows)-1} verified labels.")

    def _load_auto_labels(self):
        rows = []
        with open(self.auto_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                rows.append(row)
        return rows

    def _verify_one(self, image_path, predicted_label):
        img = cv2.imread(str(image_path))
        if img is None:
            return "skip"

        display = img.copy()
        cv2.putText(
            display,
            f"Predicted: {predicted_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Verify", display)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("c"):
                return "cat"
            elif key == ord("n"):
                return "not_cat"
            elif key == ord("s"):
                return "skip"
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return None

# CLEAN STAGE
class CleanStage(Stage):
    def __init__(
        self,
        raw_dir: Path,
        verified_labels_csv: Path,
        cleaned_dir: Path,
        cleaned_labels_csv: Path,
        log_csv: Path,
        min_width=64,
        min_height=64,
    ):
        super().__init__("Clean")
        self.raw_dir = Path(raw_dir)
        self.verified_labels_csv = Path(verified_labels_csv)
        self.cleaned_dir = Path(cleaned_dir)
        self.cleaned_labels_csv = Path(cleaned_labels_csv)
        self.log_csv = Path(log_csv)
        self.min_width = min_width
        self.min_height = min_height

    def requires(self):
        return [self.raw_dir, self.verified_labels_csv]

    def produces(self):
        return [self.cleaned_dir, self.cleaned_labels_csv, self.log_csv]

    def is_complete(self):
        return (
            self.cleaned_dir.exists()
            and self.cleaned_labels_csv.exists()
            and self.log_csv.exists()
        )

    def run(self):
        logger.info("CleanStage started.")
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

        kept_rows = []
        removed_rows = []

        with open(self.verified_labels_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)

            for filename, label in tqdm(reader, desc="Clean"):
                image_path = self.raw_dir / filename

                if not image_path.exists():
                    removed_rows.append([filename, "missing"])
                    continue

                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                except Exception:
                    removed_rows.append([filename, "corrupt"])
                    continue

                if width < self.min_width or height < self.min_height:
                    removed_rows.append([filename, "too_small"])
                    continue

                shutil.copy(image_path, self.cleaned_dir / filename)
                kept_rows.append([filename, label])

        self._write_cleaned_csv(kept_rows)
        self._write_log_csv(removed_rows)

        logger.info("CleanStage finished.")

    def validate(self):
        if not self.cleaned_dir.exists():
            raise RuntimeError("Cleaned directory not created.")

        if not self.cleaned_labels_csv.exists():
            raise RuntimeError("Cleaned CSV not created.")

        if not self.log_csv.exists():
            raise RuntimeError("Cleaning log CSV not created.")

        logger.info("CleanStage validation passed.")

    def _write_cleaned_csv(self, rows):
        with open(self.cleaned_labels_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(rows)

    def _write_log_csv(self, rows):
        with open(self.log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "reason"])
            writer.writerows(rows)

# PREPROCESS STAGE
class PreprocessStage(Stage):
    def __init__(
        self,
        cleaned_dir: Path,
        cleaned_csv: Path,
        processed_dir: Path,
        processed_csv: Path,
        log_csv: Path,
        target_size=(224, 224),
        blur_threshold=100.0,
    ):
        super().__init__("Preprocess")
        self.cleaned_dir = Path(cleaned_dir)
        self.cleaned_csv = Path(cleaned_csv)
        self.processed_dir = Path(processed_dir)
        self.processed_csv = Path(processed_csv)
        self.log_csv = Path(log_csv)
        self.target_size = target_size
        self.blur_threshold = blur_threshold

    def requires(self):
        return [self.cleaned_dir, self.cleaned_csv]

    def produces(self):
        return [self.processed_dir, self.processed_csv, self.log_csv]

    def is_complete(self):
        return (
            self.processed_dir.exists()
            and self.processed_csv.exists()
            and self.log_csv.exists()
        )

    def run(self):
        logger.info("PreprocessStage started.")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        kept_rows = []
        log_rows = []

        with open(self.cleaned_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)

            for filename, label in tqdm(reader, desc="Preprocess"):
                image_path = self.cleaned_dir / filename

                img = cv2.imread(str(image_path))
                if img is None:
                    log_rows.append([filename, "unreadable"])
                    continue

                blur_score = self._blur_score(img)

                padded = self._letterbox(img, self.target_size)
                denoised = cv2.fastNlMeansDenoisingColored(
                    padded, None, 10, 10, 7, 21
                )

                normalized = denoised.astype(np.float32) / 255.0
                out_path = self.processed_dir / filename

                cv2.imwrite(
                    str(out_path),
                    (normalized * 255).astype(np.uint8)
                )

                kept_rows.append([filename, label])

                if blur_score < self.blur_threshold:
                    log_rows.append([filename, "blurry"])

        self._write_processed_csv(kept_rows)
        self._write_log_csv(log_rows)

        logger.info("PreprocessStage finished.")

    def validate(self):
        if not self.processed_dir.exists():
            raise RuntimeError("Processed directory missing.")

        if not self.processed_csv.exists():
            raise RuntimeError("Processed CSV missing.")

        if not self.log_csv.exists():
            raise RuntimeError("Preprocess log missing.")

        with open(self.processed_csv, "r") as f:
            rows = list(csv.reader(f))

        if len(rows) <= 1:
            raise RuntimeError("Processed CSV is empty.")

        logger.info("PreprocessStage validation passed.")

    # Helpers
    def _blur_score(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _letterbox(self, img, target_size):
        h, w = img.shape[:2]
        th, tw = target_size

        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (nw, nh))

        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        top = (th - nh) // 2
        left = (tw - nw) // 2

        canvas[top:top + nh, left:left + nw] = resized
        return canvas

    def _write_processed_csv(self, rows):
        with open(self.processed_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(rows)

    def _write_log_csv(self, rows):
        with open(self.log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "note"])
            writer.writerows(rows)


# SPLIT STAGE
class SplitStage(Stage):
    def __init__(
        self,
        processed_dir: Path,
        processed_csv: Path,
        split_dir: Path,
        train_csv: Path,
        val_csv: Path,
        test_csv: Path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    ):
        super().__init__("Split")
        self.processed_dir = Path(processed_dir)
        self.processed_csv = Path(processed_csv)
        self.split_dir = Path(split_dir)

        self.train_dir = self.split_dir / "train"
        self.val_dir = self.split_dir / "val"
        self.test_dir = self.split_dir / "test"

        self.train_csv = Path(train_csv)
        self.val_csv = Path(val_csv)
        self.test_csv = Path(test_csv)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def requires(self):
        return [self.processed_dir, self.processed_csv]

    def produces(self):
        return [
            self.train_dir,
            self.val_dir,
            self.test_dir,
            self.train_csv,
            self.val_csv,
            self.test_csv,
        ]

    def is_complete(self):
        return (
            self.train_dir.exists()
            and self.val_dir.exists()
            and self.test_dir.exists()
            and self.train_csv.exists()
            and self.val_csv.exists()
            and self.test_csv.exists()
        )

    def run(self):
        logger.info("SplitStage started.")

        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        filenames = []
        labels = []

        with open(self.processed_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                filenames.append(row[0])
                labels.append(row[1])

        X_train, X_temp, y_train, y_temp = train_test_split(
            filenames,
            labels,
            test_size=(1 - self.train_ratio),
            stratify=labels,
            random_state=self.seed,
        )

        rel_val_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - rel_val_ratio),
            stratify=y_temp,
            random_state=self.seed,
        )

        self._copy_and_write(X_train, y_train, self.train_dir, self.train_csv)
        self._copy_and_write(X_val, y_val, self.val_dir, self.val_csv)
        self._copy_and_write(X_test, y_test, self.test_dir, self.test_csv)

        logger.info("SplitStage finished.")

    def validate(self):
        logger.info("SplitStage validation passed.")

    def _copy_and_write(self, filenames, labels, out_dir, csv_path):
        rows = []
        for f, label in zip(filenames, labels):
            shutil.copy(self.processed_dir / f, out_dir / f)
            rows.append([f, label])

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(rows)
