import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load the trained model
model = load_model('intel_image.h5')

# Define class names
class_names = ['glacier', 'mountain', 'buildings', 'street', 'forest', 'sea']

# Define the function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((150, 150))  # Adjust size according to your model input
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict class
def predict_image(img):
    img_preprocessed = preprocess_image(img)
    predictions = model.predict(img_preprocessed)
    predicted_class_index = np.argmax(predictions, axis=1)
    return class_names[predicted_class_index[0]], predictions[0]

# Define the Streamlit app
st.title("Image Classification with Your Model")
st.write("Upload an image to get a prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    predicted_class, prediction_probs = predict_image(img)
    
    # Display the result
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Class: {predicted_class}')
    st.write(f'Prediction Probabilities: {prediction_probs}')
