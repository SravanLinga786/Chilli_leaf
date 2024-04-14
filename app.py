import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown
app = Flask(__name__)

# Define the file ID of your model.h5 file on Google Drive
file_id = '1s1OdDwNslBdfPzZFHd4E1tyRvx5Sd_Z7'

# Define the URL to download the model from Google Drive
model_url = f'https://drive.google.com/uc?id={file_id}'

# Define the path where you want to save the downloaded model
model_path = 'model.h5'

# Download the model from Google Drive
gdown.download(model_url, model_path, quiet=False)

print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Anthracnose', 1: 'Gemini', 2: 'Healthy', 3: 'Leaf_curl', 4: 'Leaf_spot', 5: 'Powdery', 6: 'Rust'}


def getResult(image_path):
    img = load_img(image_path, target_size=(225,225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predictions=getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    return None

@app.route('/diseases')
def diseases():
    # Logic for rendering the diseases page
    return render_template('diseases.html')
@app.route('/prevention')
def prevention():
    # Logic for rendering the diseases page
    return render_template('Prevention.html')

@app.route('/about')
def about():
    # Logic for rendering the diseases page
    return render_template('about.html')

'''@app.route('/login')
def login():
    # Logic for rendering the diseases page
    return render_template('Login.html')'''


if __name__ == '__main__':
    app.run(debug=True)
