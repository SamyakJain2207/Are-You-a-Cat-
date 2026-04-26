from pathlib import Path
import tensorflow as tf
import pandas as pd
import mlflow

# Paths
WORK_DIR = Path("data/work")
PROCESSED_DIR = WORK_DIR / "processed"

TRAIN_CSV = WORK_DIR / "train.csv"
VAL_CSV = WORK_DIR / "val.csv"
TEST_CSV = WORK_DIR / "test.csv"

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "cat_classifier.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15


# Utility: CSV -> tf.data
def create_dataset(csv_path, shuffle=True):
    df = pd.read_csv(csv_path)

    file_paths = df["filename"].apply(
        lambda x: str(PROCESSED_DIR / x)
    ).values

    labels = df["label"].apply(
        lambda x: 1 if x == "cat" else 0
    ).values

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label


# Model
def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

# ── NEW: logs metrics to MLflow after every epoch ──
class MLflowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metrics(
                {
                    "train_loss":     logs.get("loss"),
                    "train_accuracy": logs.get("accuracy"),
                    "val_loss":       logs.get("val_loss"),
                    "val_accuracy":   logs.get("val_accuracy"),
                },
                step=epoch,
            )


# ── UPDATED: now accepts hyperparams as arguments ──
def main(lr=1e-4, dropout=0.3, epochs=EPOCHS):
    mlflow.set_experiment("cat-not-cat-classifier")

    # run_name makes it easy to identify runs in the UI
    run_name = f"lr={lr}_dropout={dropout}_epochs={epochs}"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params({
            "img_size":       IMG_SIZE,
            "batch_size":     BATCH_SIZE,
            "epochs":         epochs,
            "learning_rate":  lr,
            "base_model":     "MobileNetV2",
            "dropout":        dropout,
            "optimizer":      "Adam",
        })

        print("Loading datasets...")
        train_ds = create_dataset(TRAIN_CSV, shuffle=True)
        val_ds   = create_dataset(VAL_CSV,   shuffle=False)
        test_ds  = create_dataset(TEST_CSV,  shuffle=False)

        # ── rebuild model with the dropout value passed in ──
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )
        base_model.trainable = False

        inputs  = tf.keras.Input(shape=(224, 224, 3))
        x       = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x       = base_model(x, training=False)
        x       = tf.keras.layers.GlobalAveragePooling2D()(x)
        x       = tf.keras.layers.Dropout(dropout)(x)       # ← uses the argument
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),          # ← uses the argument
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(MODEL_PATH),
                monitor="val_loss",
                save_best_only=True,
            ),
            MLflowCallback(),
        ]

        print(f"Training with lr={lr}, dropout={dropout}...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )

        loss, acc = model.evaluate(test_ds)
        mlflow.log_metrics({"test_loss": loss, "test_accuracy": acc})

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.keras.log_model(model, artifact_path="keras-model")

        print(f"\nTest Accuracy: {acc:.4f} | Run: {run_name}")


if __name__ == "__main__":
    main()