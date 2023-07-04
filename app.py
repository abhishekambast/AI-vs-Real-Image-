
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL
 








import streamlit as st
from tensorflow.keras.models import load_model






@st.cache(allow_output_mutation=True)
def load_curr_model():
    with st.spinner('Model is being loaded..'):
        curr_model = load_model('AIvsREAL.h5')
    return curr_model

st.title("AI generated Images vs Real Images")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://zh-prod-1cc738ca-7d3b-4a72-b792-20bd8d8fa069.storage.googleapis.com/s3fs-public/inline-images/AI-human-heads.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 



file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
st.markdown("\n\n\Dataset used for training this model **:blue[https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images]**.")
st.write("\n\n By Abhishek Ambast.")
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    
        size = (126,126)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        
        
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, load_curr_model())
    
    st.write(prediction)
    
    if(np.argmax(prediction)==1):
        st.write("This is a Real Image")
    else:
        st.write("This is an AI generated Image")
    
    
st.markdown("Link to Github repository **:blue[https://github.com/abhishekambast/AI-vs-Real-Image-.git]**.")
