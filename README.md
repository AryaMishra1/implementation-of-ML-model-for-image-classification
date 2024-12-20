# implementation-of-ML-model-for-image-classification

#Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
from PIL import Image
import os

Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

Load the MobileNetV2 model pre-trained on ImageNet
base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

Add a custom classification head
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(10, activation='softmax')(x)

Create the model
model = keras.Model(inputs=base_model.input, outputs=x)

Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')

Create a Streamlit app
st.title('Image Classification App')
st.write('Upload an image to classify it')

Create a file uploader
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'png', 'jpeg'])

Create a button to classify the image
if st.button('Classify'):
    # Preprocess the uploaded image
    img = Image.open(uploaded_file)
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
   
    # Make predictions on the uploaded image
    predictions = model.predict(img[np.newaxis, ...])
    predicted_class = np.argmax(predictions)
   
    # Display the classification result
    st.write(f'Predicted class: {predicted_class}')
