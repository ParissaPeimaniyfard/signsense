import streamlit as st
import requests
from PIL import Image

# Function to make API request
def predict_letter(uploaded_file):
    # Send request to FASTapi API
    response = requests.post('https://signsense-api-viakkexzvq-ew.a.run.app/predict', files={'image_file': uploaded_file})
    predicted_letter = response.json()['prediction']
    return predicted_letter

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