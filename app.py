import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import PIL

# Load the trained model
model = load_model('intel_image.h5')

# Define the function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))  # Adjust size according to your model input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Define the Streamlit app
st.title("Image Classification with Your Model")
st.write("Upload an image to get a prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    img = PIL.Image.open(uploaded_file)
    img_preprocessed = preprocess_image(img)
    
    # Predict the class
    predictions = model.predict(img_preprocessed)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Display the result
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Class: {predicted_class[0]}')
