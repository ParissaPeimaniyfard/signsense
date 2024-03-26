from fastapi import FastAPI, UploadFile, File
from tensorflow.keras import models
import numpy as np
from PIL import Image
from google.cloud import storage
from package_folder.train_model import ALPHABET
import cv2
from PIL import Image
import io



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


@app.post("/predict_video")
async def predict_video(video_file: UploadFile = File(...)):
    # Read the uploaded video file
    video_bytes = await video_file.read()

    # Initialize an empty predicted word
    predicted_word = ""

    # Open the video stream
    cap = cv2.VideoCapture(io.BytesIO(video_bytes))

    # Read each frame from the video
    while True:
        ret, frame = cap.read()

        # If there are no more frames, break the loop
        if not ret:
            break

        # Convert the frame to PIL Image format
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize the image to match your model input shape (assuming 50x50)
        frame_pil = frame_pil.resize((50, 50))

        # Convert the PIL Image to numpy array
        frame_np = np.array(frame_pil)

        # Reshape the numpy array to match the model input shape
        frame_np = frame_np.reshape((-1, 50, 50, 3))

        # Perform prediction using your model
        prediction = model_good.predict(frame_np)

        # Get the predicted index
        predicted_idx = np.argmax(prediction)

        # Map the predicted index to the corresponding letter
        predicted_letter = [letter for letter, indic in ALPHABET.items() if indic == predicted_idx]

        # Append the predicted letter to the predicted word
        predicted_word += predicted_letter[0]  # Assuming only one prediction per frame

    # Release the video capture object
    cap.release()

    # Return the predicted word
    return {"predicted_word": predicted_word}
