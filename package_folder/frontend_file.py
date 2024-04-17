import streamlit as st
import requests
from PIL import Image
import cv2

# Function to make API request
def predict_letter(uploaded_file):
    # Send request to FASTapi API
    response = requests.post('https://signvideo-viakkexzvq-ew.a.run.app/predict', files={'image_file': uploaded_file})
    predicted_letter = response.json()['prediction']
    return predicted_letter

# Function to make API request for video prediction
def predict_video(uploaded_file):
    # Send request to FastAPI endpoint for video prediction
    response = requests.post('https://signvideo-viakkexzvq-ew.a.run.app/predict_video', files={'video_file': uploaded_file})
    predicted_word = response.json()['predicted_word']
    return predicted_word



# Streamlit UI
st.title('Sign Language Letter Prediction')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    uploaded_file.seek(0)

    # Button to trigger prediction
    if st.button('Predict'):
        # Make API request
        predicted_letter = predict_letter(uploaded_file)
        st.write(f'Predicted Letter: {predicted_letter}')


# Streamlit UI
st.title('Sign Language Word Prediction')

# File uploader for video
uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

# video_file = open(uploaded_video)
# video_bytes = video_file.read()

if uploaded_video is not None:
    # Display uploaded video

    # st.video(video_bytes)

    st.video(uploaded_video, format='video/mp4')

    # Button to trigger video prediction
    if st.button('Predict Video'):
        # Make API request for video prediction
        predicted_word = predict_video(uploaded_video)
        st.write(f'Predicted Word: {predicted_word}')
