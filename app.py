import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained CNN model
model = tf.keras.models.load_model(r".\model\traffic_sign_model.h5")
IMG_SIZE = (32, 32)

# Load Class Labels from CSV
csv_file = r".\dataset\Indian-Traffic Sign-Dataset\traffic_sign.csv"
df = pd.read_csv(csv_file)
CLASS_LABELS = {row["ClassId"]: row["Name"] for _, row in df.iterrows()}

# Function to Predict Traffic Sign
def predict_traffic_sign(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    return CLASS_LABELS.get(class_id, "Unknown Sign")

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No selected file")
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        prediction = predict_traffic_sign(filepath)
        return render_template("index.html", uploaded_image=filepath, prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)