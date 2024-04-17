from fastapi import FastAPI, UploadFile, File
from tensorflow.keras import models
import tensorflow as tf
import numpy as np
from PIL import Image
from google.cloud import storage
from package_folder.train_model import ALPHABET
import cv2
from PIL import Image
import io
# import pickle
import shutil
import tempfile



# client = storage.Client()
# bucket = client.bucket('raw-data-signsense')
# blob = bucket.get_blob('model-params.h5')
# blob.download_to_filename('/tmp/model-params.h5')
# model_good = models.load_model('/tmp/model-params.h5')

app = FastAPI()

# Load the image classification model
model = tf.keras.models.load_model('models/dummy_model_size100_tensflow_version10.h5')

## Root Endpoint (Landing Page)
@app.get("/")
def root():
    return {'greeting': "Hello User"}

## Predict Endpoint where model is located
# ("/predict") specifies the URL specificity
@app.post("/predict")
def predict(image_file: UploadFile):
    image = Image.open(image_file.file)
    image = image.resize((100, 100))
    img = np.array(image)
    img = img.reshape((-1, 100, 100, 3))
    prediction = model.predict(img)
    predicted_idx = np.argmax(prediction)
    predicted_letter = [letter for letter, indic in ALPHABET.items() if indic == predicted_idx]
    # print(f'probability: {prediction[0][predicted_idx]*100}')
    return {"prediction": predicted_letter}


@app.post("/predict_video")
async def predict_video(video_file: UploadFile = File(...)):
    # Create a temporary directory to save the video file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = f"{temp_dir}/temp_video.mp4"

        # Save the uploaded video to the temporary directory
        with open(temp_video_path, "wb") as temp_video_file:
            shutil.copyfileobj(video_file.file, temp_video_file)


        # Initialize an empty predicted word
        predicted_word = ""

        # Open the video stream
        cap = cv2.VideoCapture(temp_video_path)

    # Read each frame from the video
    while True:
        ret, frame = cap.read()

        # If there are no more frames, break the loop
        if not ret:
            break

        # Convert the frame to PIL Image format
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize the image to match your model input shape (assuming 50x50)
        frame_pil = frame_pil.resize((100, 100))

        # Convert the PIL Image to numpy array
        frame_np = np.array(frame_pil)

        # Reshape the numpy array to match the model input shape
        frame_np = frame_np.reshape((-1, 100, 100, 3))

        # Perform prediction using your model
        prediction = model.predict(frame_np)

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
