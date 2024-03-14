import sys
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from tensorflow.keras import Sequential, layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import layers, optimizers, callbacks
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.models import load_model
from google.cloud import storage

ALPHABET = {
    'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12,
    "O":13, "P":14, 'Q':15, 'R':16, 'S':17, 'T':18, 'U':19,
    "V":20, 'W':21, 'X':22, "Y":23, 'del':24, 'nothing':25, 'space':26
}

def load_asl_data(max_imgs=0):
    client = storage.Client()
    bucket = client.bucket('raw-data-signsense')
    imgs = []
    labels = []
    for (cl, i) in ALPHABET.items():
        print(f'Letter {cl}')
        idx = 0
        for blob in bucket.list_blobs():
            if f'asl_alphabet_train/asl_alphabet_train/{cl}' in blob.name:
                idx += 1
                if max_imgs and idx > max_imgs:
                    break
                image = Image.open(blob.open('rb'))
                image = image.resize((50, 50))
                imgs.append(np.array(image))
                labels.append(i)

    X = np.array(imgs)
    num_classes = len(set(labels))
    y = to_categorical(labels, num_classes)

    # Finally we shuffle:
    p = np.random.permutation(len(X))
    X, y = X[p], y[p]

    first_split = int(len(imgs) /6.)
    second_split = first_split + int(len(imgs) * 0.2)
    X_test, X_val, X_train = X[:first_split], X[first_split:second_split], X[second_split:]
    y_test, y_val, y_train = y[:first_split], y[first_split:second_split], y[second_split:]

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes

def good_model():
  input_shape= (50,50,3)
  model = Sequential()

  model.add(layers.Rescaling(1./255, input_shape = input_shape))
  model.add(layers.Conv2D(filters = 32, kernel_size = (7,7), activation="relu", padding = "same"))
  model.add(BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = "same") )


  model.add(layers.Conv2D(filters = 64, kernel_size = (5,5), activation="relu", padding = "same"))
  model.add(BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = "same") )


  model.add(layers.Conv2D(filters = 128, kernel_size = (5,5), activation="relu", padding = "same"))
  model.add(BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = "same") )

  model.add(layers.Conv2D(filters = 512, kernel_size = (3,3), activation="relu", padding = "same"))
  model.add(BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = "same"))

  model.add(layers.Conv2D(filters = 1024, kernel_size = (3,3), activation="relu", padding = "same"))
  model.add(BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = "same"))

  model.add(layers.Flatten())

  # Here we flatten our data
  model.add(layers.Dense(512, activation="relu"))
  model.add(layers.Dropout(0.2))
  model.add(BatchNormalization())

  model.add(layers.Dense(128, activation="relu"))
  model.add(layers.Dropout(0.2))
  model.add(BatchNormalization())

  model.add(layers.Dense(64, activation="relu"))
  model.add(layers.Dropout(0.2))
  model.add(BatchNormalization())

  # prediction layer
  model.add(layers.Dense(27, activation="softmax"))

  # compiling model
  adam = optimizers.Adam(learning_rate = 0.01)
  model.compile(loss='categorical_crossentropy',
                optimizer= adam,
                metrics=['accuracy'])
  return model


if __name__ == '__main__':
    max_imgs = 0
    if len(sys.argv) > 1:
        max_imgs = int(sys.argv[1])
    model_good = good_model()
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_asl_data(max_imgs)
    modelCheckpooint = callbacks.ModelCheckpoint("good_model.h5", monitor="val_loss", verbose=0, save_best_only=True)
    # LRreducer = callbacks.ReduceLROnPlateau(monitor="val_loss", factor = 0.1, patience=3, verbose=1, min_lr=0)
    EarlyStopper = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)
    history = model_good.fit(
        X_train, y_train,
        epochs=20,
        validation_data= (X_val, y_val),
        callbacks = [modelCheckpooint, EarlyStopper]
    )
    tmp_path = '/tmp/model-params.h5'
    model_good.save(tmp_path)
    client = storage.Client()
    bucket = client.bucket('raw-data-signsense')
    blob = bucket.blob('model-params.h5')
    blob.upload_from_filename(tmp_path)

