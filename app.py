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
import sys
import keras

# ---- Keras compatibility patch ----
sys.modules['keras.src'] = keras
sys.modules['keras.src.models'] = keras.models
sys.modules['keras.src.layers'] = keras.layers
sys.modules['keras.src.utils'] = keras.utils
sys.modules['keras.src.backend'] = keras.backend
# ----------------------------------
# ---------------------------
# CREATE FLASK APP
# ---------------------------
app = Flask(__name__, static_folder="static")
CORS(app)
# Create static folder if not exists
if not os.path.exists("static"):
    os.makedirs("static")
# ---------------------------
# LOAD MODEL
# ---------------------------
MODEL_PATH = "dr_efficientnet_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded")
except Exception as e:
    print("Model failed to load:", e)
    model = None
class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]
# ---------------------------
# GRADCAM FUNCTION
# ---------------------------
def generate_gradcam(model, image, class_index):
    # Ensure float32
    image = tf.cast(image, tf.float32)
    # Make sure batch dimension exists
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)
    # IMPORTANT FIX → clone=False
    gradcam = GradcamPlusPlus(
        model,
        model_modifier=ReplaceToLinear(),
        clone=False
    )
    # Pass class index correctly
    score = CategoricalScore([class_index])
    # Generate CAM
    cam = gradcam(score, image)
    heatmap = cam[0]
    # Normalize safely
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    return heatmap
@app.route('/', methods=['GET'])
def home():
    return "Backend is running!"
# ---------------------------
# PREDICTION ROUTE
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    # Read image
    original_img = Image.open(file).convert("RGB")
    original_img = original_img.resize((224, 224))
    # Convert to numpy
    original_array = np.array(original_img)
    # Prepare for model
    img = np.expand_dims(original_array, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    # Prediction
    preds = model.predict(img)
    # Convert logits to probabilities
    probs = tf.nn.softmax(preds[0]).numpy()
    predicted_index = int(np.argmax(probs))
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(probs))
    # Generate GradCAM
    heatmap = generate_gradcam(model, img, predicted_index)
    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (original_array.shape[1], original_array.shape[0])
    )
    # Convert heatmap to color
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # IMPOMRTANT FIX → use original_array (numpy), not original_img
    overlay = cv2.addWeighted(original_array, 0.6, heatmap, 0.4, 0)
    # Save image
    gradcam_path = os.path.join(app.static_folder, "gradcam.jpg")
    cv2.imwrite(gradcam_path, overlay)
    return jsonify({
        "prediction": predicted_label,
        "confidence": confidence,
        "gradcam_url": "http://127.0.0.1:5000/static/gradcam.jpg"
    })
# ---------------------------
# RUN SERVER
# ---------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
