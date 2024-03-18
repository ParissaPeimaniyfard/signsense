
import streamlit as st
import requests
from PIL import Image
import io

st.title("Sign sense app")
st.write("Upload your image")

# File uploader widget
uploaded_file = st.file_uploader("Import your image", type=["jpg", "jpeg"])
# print(uploaded_file)
# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    st.image(image, caption='Uploaded Image.', use_column_width=True)



def predict_image(image):

    # image_data = image.read()
    # API endpoint for prediction
    api_endpoint = 'https://signsense-api-viakkexzvq-ew.a.run.app/predict'

    # image= Image.open(image)
    # Prepare the image data

    files = {'image_file': image}
    # print(img_bytes.getvalue())
    # test= f'http://signsenseapi2-viakkexzvq-ew.a.run.app/predict/image_file={image.filename};type=image/jpeg'
    # print(test)
    # Send the image to the API endpoint for prediction

    # response = requests.get(test)
    response = requests.post(api_endpoint, files=files)
    print(response.status_code)
    print(response)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the prediction result
        prediction = response.json()
        # st.write(prediction)
        # print(prediction)
        return prediction
    else:
        return None


# Button to trigger prediction
if st.button('Predict'):
    # image_testing= Image.open('B_test.jpg')
    # Send the uploaded image to the API for prediction
    prediction_result = predict_image(uploaded_file)

    # Display the prediction result
    if prediction_result is not None:
        st.write("Prediction:", prediction_result)
    else:
        st.write("Error: Failed to get prediction from API")
