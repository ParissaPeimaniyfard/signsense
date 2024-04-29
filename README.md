
## ASL Sign Language Recognition
Overview
This project aims to develop a machine learning model for American Sign Language (ASL) recognition, enabling the prediction of letters and words from images and videos of ASL gestures. The system utilizes various techniques including convolutional neural networks (CNNs), transfer learning, data augmentation, and landmark detection for accurate recognition of ASL gestures.

### Dataset to be used: Kaggle Sign Language MNIST:https://www.kaggle.com/datasets/grassknoted/asl-alphabet
The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.

The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.

Each training and test case represents a label (0-27) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).

For some modeles 3 classes for SPACE, DELETE and NOTHING were removed to check if the model can work better.

### Features
Classification of ASL letters from images.
Prediction of ASL words from videos composed of frames.
Utilization of CNN architectures for image classification.
Transfer learning with VGG16 and YOLOv8 models for improved performance.
Data augmentation techniques to enhance model robustness.
Landmark detection using MediaPipe models to detect hand gestures.
### Usage
#### Training the Model
Data Preparation: Prepare a dataset of ASL images and videos for training. Ensure that the dataset includes a diverse range of ASL gestures and is labeled accordingly.

#### Model Training:
Train the models using the provided scripts or notebooks. Experiment with different CNN architectures, transfer learning techniques, and data augmentation methods to optimize performance.

#### Modeling Environment
The models for this project were developed using Google Colab, a cloud-based Jupyter notebook environment provided by Google. Google Colab was chosen for its access to GPU/TPU resources and generous RAM allocation, which proved invaluable for training deep neural networks and handling large datasets.

### Running the API
_Installation:_ Install the necessary dependencies by running *"pip install -r requirements.txt"*.

_Starting the Server:_ Launch the FastAPI server by running *"uvicorn package_folder.api_file:app --reload"*.

_API Usage:_ Use the provided API endpoints to interact with the ASL recognition system. Refer to the API documentation for detailed instructions on endpoint usage.

_Deployment:_ Deploy the trained model and FastAPI server to a cloud platform.

_API Documentation:_ The API provides the following endpoints:
      _/predict:_ Accepts an image or video file containing ASL gestures and returns the predicted letter or word.
