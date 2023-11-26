from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('../models/final_model')
animals = ["bat", "bee", "cat", "duck", "elephant", "lion", "octopus", "rabbit", "snail", "whale"]

# Preprocess the uploaded image file to the format expected by the model.
def preprocess_image(image_bytes):
    # Load the image
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((64, 64))  # Resize to the expected input size

    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 64, 64, 1))  # Reshape for the model

    return img_array

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Get the file from the request
        file = request.files['file']

        # Preprocess the image and predict
        image_bytes = file.read()
        preprocessed_image = preprocess_image(image_bytes)
        prediction = model.predict(preprocessed_image)

        # Find the index of the max probability
        max_index = np.argmax(prediction[0])
        most_likely_class = animals[max_index]

        # Return both the raw probabilities and the most likely class
        return jsonify({
            "prediction": prediction[0].tolist(), 
            "most_likely_class": most_likely_class,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)