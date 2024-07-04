import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('model_for_nuclei.h5')

# Constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

st.title("Nuclei Segmentation")
st.write("Upload an image and get the segmented output.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Preprocess the image
    image_resized = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    image_resized = image_resized[:,:,:IMG_CHANNELS]
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension

    # Make prediction
    preds = model.predict(image_resized)
    preds = (preds > 0.5).astype(np.uint8)

    # Display the output using plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[1].imshow(np.squeeze(preds[0]), cmap='viridis')
    ax[1].set_title('Segmented Output')
    ax[2].imshow(image)
    ax[2].imshow(np.squeeze(preds[0]), cmap='gray', alpha=0.5)
    ax[2].set_title('Overlay')
    for a in ax:
        a.axis('off')
    
    st.pyplot(fig)
