import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL

# Load the trained model
model = load_model('intel_image.h5')

# Define class names
class_names = ['mountain', 'glacier', 'sea', 'forest', 'street', 'buildings']

# Define the function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((150, 150))  # Adjust size according to your model input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image if required
    return img_array

# Define the Streamlit app
st.title("Image Classification with Your Model")
st.write("Upload an image to get a prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = PIL.Image.open(uploaded_file)
    img_preprocessed = preprocess_image(img)
    
    # Predict the class
    predictions = model.predict(img_preprocessed)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class_index[0]]
    
    # Display the result
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Class: {predicted_class_name}')
