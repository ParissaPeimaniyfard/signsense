import streamlit as st
import requests
from PIL import Image

# Function to make API request
def predict_word(uploaded_file):
    # Send request to FASTapi API
    response = requests.post('https://signsense-api-viakkexzvq-ew.a.run.app/predict_video', files={'video_file': uploaded_file})
    predicted_word = response.json()['prediction']
    return predicted_word

# Streamlit UI
st.title('Sign Language Letter Prediction')

uploaded_file = st.file_uploader("Upload a video", type=["jpg", "png", "jpeg", "mp4"])

if uploaded_file is not None:

    # Button to trigger prediction
    if st.button('Predict'):
        # Make API request
        predicted_word = predict_word(uploaded_file)
        st.write(f'Predicted Word: {predicted_word}')
