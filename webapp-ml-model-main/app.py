from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('../models/final_model')
animals = ["bat", "bee", "cat", "duck", "elephant", "lion", "octopus", "rabbit", "snail", "whale"]

def preprocess_image(image_bytes, image_size_v):
    # Load the image from the bytes
    img = Image.open(io.BytesIO(image_bytes))

    # Convert the image to grayscale
    # 'L' mode means 8-bit pixels, black and white
    img = img.convert('L')

    # Resize the image to the specified size
    img = img.resize((image_size_v, image_size_v))
    img_array = np.array(img)
    img_array = img_array.astype('uint8')

    # Reshape the array for the model
    img_array = img_array.reshape((1, image_size_v, image_size_v, 1))

    return img_array

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Get the file from the request
        file = request.files['file']

        image_size_v = 64

        # Preprocess the image and predict
        image_bytes = file.read()
        preprocessed_image = preprocess_image(image_bytes, image_size_v)
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