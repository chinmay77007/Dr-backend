from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import os
import uuid
import requests

# ---------------------------
# CREATE FLASK APP
# ---------------------------
app = Flask(__name__, static_folder="static")
CORS(app)

if not os.path.exists("static"):
    os.makedirs("static")

# ---------------------------
# LOAD MODEL
# Downloads from Google Drive on first startup
# ---------------------------

MODEL_PATH = "model.keras"
MODEL_URL  = os.environ.get("MODEL_URL", "")

def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already present.")
        return
    if not MODEL_URL:
        print("No MODEL_URL set.")
        return
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded.")

download_model()

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print("Model failed to load:", e)
    model = None

class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMG_SIZE = 224  # old model uses 224px


# ---------------------------
# GRADCAM
# ---------------------------

def generate_gradcam(model, image, class_index):
    image = tf.cast(image, tf.float32)
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)

    gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=False)
    score   = CategoricalScore([class_index])
    cam     = gradcam(score, image)
    heatmap = cam[0]

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    return np.uint8(255 * heatmap)


# ---------------------------
# ROUTES
# ---------------------------

@app.route("/", methods=["GET"])
def home():
    return "DR Backend is running!"


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Read and resize image
    original_img   = Image.open(file).convert("RGB")
    original_img   = original_img.resize((IMG_SIZE, IMG_SIZE))
    original_array = np.array(original_img)

    # Prepare for model — old model uses efficientnet preprocess_input
    img = np.expand_dims(original_array, axis=0).astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    # Predict
    preds           = model.predict(img)
    probs           = tf.nn.softmax(preds[0]).numpy()
    predicted_index = int(np.argmax(probs))
    predicted_label = class_names[predicted_index]
    confidence      = float(np.max(probs))

    # GradCAM
    try:
        heatmap  = generate_gradcam(model, img, predicted_index)
        heatmap  = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap  = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        orig_bgr = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
        overlay  = cv2.addWeighted(orig_bgr, 0.6, heatmap, 0.4, 0)

        filename  = f"gradcam_{uuid.uuid4().hex}.jpg"
        filepath  = os.path.join("static", filename)
        cv2.imwrite(filepath, overlay)

        base_url  = request.host_url.rstrip("/")
        image_url = f"{base_url}/static/{filename}"

    except Exception as e:
        print("GradCAM failed:", e)
        image_url = None

    return jsonify({
        "prediction": predicted_label,
        "confidence": confidence,
        "gradcam_url": image_url,
    })


# ---------------------------
# RUN
# ---------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
