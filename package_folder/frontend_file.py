import streamlit as st
import requests
st.title("Sign sense app")

st.write("Upload your image")

###### Needs to be adapted 
# Creating four sliders

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

# TEST LINE
response=requests.get(f"https://mvp-viakkexzvq-ew.a.run.app/predict?x1={value1}&x2={value2}&x3={value3}&x4={value4}").json()
st.write("The image represents the letter", str(response['prediction']))