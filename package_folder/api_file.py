from fastapi import FastAPI, UploadFile
from tensorflow.keras import models
import numpy as np
from PIL import Image
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('raw-data-signsense')
blob = bucket.get_blob('model-params.h5')
model_good = models.load_model(blob.open('rb'))
    
app = FastAPI()

## Root Endpoint (Landing Page)
@app.get("/")
def root(): 
    return {'greeting': "Hello User"}

## Predict Endpoint where model is located
# ("/predict") specifies the URL specificity
@app.get("/predict")
def predict(image_file: UploadFile):
    image = Image.open(image_file)
    image = image.resize((50, 50))
    img = np.array(image)
    img = img.reshape((-1, 50, 50, 3))
    prediction = model_good.predict(img)
    return {"prediction": float(prediction[0])}
        