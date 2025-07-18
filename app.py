import streamlit as st 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.preprocessing import image 
from PIL import Image 

from tensorflow.keras.models import load_model


model = load_model('pneumonia_pred_model.keras')

class_name = ['Normal','Pneumonia']


st.markdown(
    """
    <p style="font-size: 40px; color: black;">
        Pneumonia Detection from Chest X-Ray
    </p>
    """,
    unsafe_allow_html=True
)

st.write('Upload a cheast X-ray image to predict whether it shows signs of pneumonia disease')

uploaded_file = st.file_uploader('Choose and X-ray Image' , type = ['jpg','png','jpeg'])


if uploaded_file is not None : 
     img = Image.open(uploaded_file).convert('RGB')
     st.image(img,caption="Image Uploaded Succesfully", use_column_width=True)
     
     # Preprocessing 
     img = img.resize((32,32)) # Resized Image
     image_array = image.img_to_array(img)/255.0 # Rescaled image
     image_array = np.expand_dims(image_array,axis=0) # Adding 1 Dimension
     

     predication = model.predict(image_array)
     confidence = float(predication[0][0])


     if confidence >= 0.5:
          st.error("Predication: Pneumonia Detected")
     else:
          st.success("Predication: No Pneumonia Detected")

