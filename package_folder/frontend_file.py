import streamlit as st
import requests
st.title("Sign sense app")

st.write("Upload your image")


# Creating four sliders
value1 = st.slider('Select a value for Sepal length', min_value=0, max_value=4, value=1, step=1)
value2 = st.slider('Select a value for Sepal width',  min_value=0, max_value=4, value=1, step=1)
value3 = st.slider('Select a value for Petal length',  min_value=0, max_value=4, value=1, step=1)
value4 = st.slider('Select a value for Petal width',  min_value=0, max_value=4, value=1, step=1)

# TEST LINE
response=requests.get(f"https://mvp-viakkexzvq-ew.a.run.app/predict?x1={value1}&x2={value2}&x3={value3}&x4={value4}").json()
st.write("The image represents the letter", str(response['prediction']))