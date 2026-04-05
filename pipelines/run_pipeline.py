from pathlib import Path

from pipelines.base import PipelineRunner
from pipelines.stages import (
    AutoLabelStage,
    VerifyStage,
    CleanStage,
    PreprocessStage,
    SplitStage
)


def main():
    # Base Directories
    RAW_DIR = Path("data/raw")
    WORK_DIR = Path("data/work")

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Stage Output Paths
    auto_csv = WORK_DIR / "auto_labels.csv"
    verified_csv = WORK_DIR / "verified_labels.csv"

    cleaned_dir = WORK_DIR / "cleaned"
    cleaned_csv = WORK_DIR / "cleaned_labels.csv"
    clean_log = WORK_DIR / "clean_log.csv"

    processed_dir = WORK_DIR / "processed"
    processed_csv = WORK_DIR / "processed_labels.csv"
    preprocess_log = WORK_DIR / "preprocess_log.csv"

    split_dir = WORK_DIR / "splits"
    train_csv = WORK_DIR / "train.csv"
    val_csv = WORK_DIR / "val.csv"
    test_csv = WORK_DIR / "test.csv"

    # Define Stages
    stages = [
        AutoLabelStage(
            raw_dir=RAW_DIR,
            output_csv=auto_csv,
            confidence_threshold=0.7,
        ),

        VerifyStage(
            raw_dir=RAW_DIR,
            auto_csv=auto_csv,
            verified_output_csv=verified_csv,
        ),

        CleanStage(
            raw_dir=RAW_DIR,
            verified_labels_csv=verified_csv,
            cleaned_dir=cleaned_dir,
            cleaned_labels_csv=cleaned_csv,
            log_csv=clean_log,
            min_width=64,
            min_height=64,
        ),

        PreprocessStage(
            cleaned_dir=cleaned_dir,
            cleaned_csv=cleaned_csv,
            processed_dir=processed_dir,
            processed_csv=processed_csv,
            log_csv=preprocess_log,
            target_size=(224, 224),
            blur_threshold=100.0,
        ),

        SplitStage(
            processed_dir=processed_dir,
            processed_csv=processed_csv,
            split_dir=split_dir,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        ),
    ]

    # Run Pipeline
    runner = PipelineRunner(stages)
    runner.run()


if __name__ == "__main__":
    main()