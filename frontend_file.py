# THIS FILE GOES TO PACKAGE_FOLDER AS SOON AS IT IS READY

import streamlit as st
import requests
st.title("Sign sense app")

st.write("Upload your image")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    

# TEST LINE
response = requests.get(f"https://signsense-api-viakkexzvq-ew.a.run.app/predict={}").json()

# TEST LINE
#response=requests.get(f"https://mvp-viakkexzvq-ew.a.run.app/predict?x1={value1}&x2={value2}&x3={value3}&x4={value4}").json()
st.write("The image represents the letter", str(response['prediction']))



params= would be the image .ipeg

wagon_cab_api_url = 'https://taxifare.lewagon.ai/predict'
response = requests.get(wagon_cab_api_url, params=params)