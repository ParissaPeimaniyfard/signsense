from fastapi import FastAPI, UploadFile
from tensorflow.keras import models
import numpy as np
import requests
from PIL import Image
from google.cloud import storage
from package_folder.train_model import ALPHABET

client = storage.Client()
bucket = client.bucket('raw-data-signsense')
blob = bucket.get_blob('model-params.h5')
blob.download_to_filename('/tmp/model-params.h5')
model_good = models.load_model('/tmp/model-params.h5')

app = FastAPI()

## Root Endpoint (Landing Page)
@app.get("/")
def root():
    return {'greeting': "Hello User"}

## Predict Endpoint where model is located
# ("/predict") specifies the URL specificity
@app.route('/predict', methods=['POST'])
def predict():
    file = requests.files['image_file']
    image = Image.open(file.stream)
    image = image.resize((50, 50))
    img = np.array(image)
    img = img.reshape((-1, 50, 50, 3))
    prediction = model_good.predict(img)
    predicted_idx = np.argmax(prediction)
    predicted_letter = [letter for letter, indic in ALPHABET.items() if indic == predicted_idx]
    return {"prediction": predicted_letter}
