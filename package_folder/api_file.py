from fastapi import FastAPI
import pickle

app = FastAPI()


## Root Endpoint (Landing Page)
@app.get("/")

def root():
    return {'greeting': "Hello User"}

## Predict Endpoint where model is located
# ("/predict") specifies the URL specificity
@app.get("/predict")
 
 # def predict() is defined as a function which takes the features as arguments
 # Features are passed by the user to the model to get the prediction
def predict(x1, x2, x3, x4):
    with open('models/best_model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    prediction = model.predict([[x1, x2, x3, x4]])
    
    return {"prediction": float(prediction[0])}
        