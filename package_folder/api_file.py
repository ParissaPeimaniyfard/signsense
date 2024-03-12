<<<<<<< HEAD
from fastapi import FastAPI, UploadFile
from tensorflow.keras import models
import numpy as np
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
@app.post("/predict")
def predict(image_file: UploadFile):
    image = Image.open(image_file.file)
    image = image.resize((50, 50))
    img = np.array(image)
    img = img.reshape((-1, 50, 50, 3))
    prediction = model_good.predict(img)
    predicted_idx = np.argmax(prediction)
    predicted_letter = [letter for letter, indic in ALPHABET.items() if indic == predicted_idx]
    return {"prediction": predicted_letter}
        
=======
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Create FastAPI app instance
app = FastAPI()

# Load the image classification model
model = tf.keras.models.load_model('models/dummy_model_100size.h5')

# Define route for root endpoint
@app.get('/')
def root():
    return {'greeting': 'hello'}

# Define route for image classification
@app.post('/predict_image')
async def predict_image(file: UploadFile = File(...)):
    # Read image data from the uploaded file
    contents = await file.read()

    # Open the image using PIL
    img = Image.open(BytesIO(contents))

    # Preprocess the image for the model
    img = img.resize((100, 100))  # Assuming your model expects 100*100 images
    img_array = np.array(img)
    img_array = img_array.reshape((-1, 100, 100, 3))   # Add batch dimension

    # Make predictions using the loaded model
    predictions = model.predict(img_array)
    predicted_indc= np.argmax(predictions)
    classes = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, "O":13, "P":14, 'Q':15, 'R':16, 'S':17, 'T':18, 'U':19,
                "V":20, 'W':21, 'X':22, "Y":23, 'del':24, 'nothing':25, 'space':26}
    predicted_letter = [letter for letter, indic in classes.items() if indic == predicted_indc]

    return {'predicted_letter': predicted_letter[0]}
>>>>>>> master
