from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# Config
# =========================
MODEL_PATH = Path("model/cat_classifier.keras")
IMG_SIZE = (224, 224)


# =========================
# Load Model (once)
# =========================
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# =========================
# Preprocess Image
# =========================
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)

    # IMPORTANT: same preprocessing as training
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    return arr


# =========================
# Predict
# =========================
def predict(image_path, model):
    processed = preprocess_image(image_path)

    prob = model.predict(processed)[0][0]

    label = "cat" if prob >= 0.5 else "not_cat"

    return {
        "label": label,
        "confidence": float(prob)
    }


if __name__ == "__main__":
    model = load_model()
    
    test_images = [
        "data/raw/gh_00001.jpg",
        "data/raw/kg1_00042.jpg",
    ]
    
    for image_path in test_images:
        result = predict(image_path, model)
        print(f"\nImage      : {image_path}")
        print(f"Label      : {result['label']}")
        print(f"Confidence : {result['confidence']:.4f}")