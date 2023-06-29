import streamlit as st
import tensorflow as tf
import streamlit as st


@st.cache(allow_output_mutation=True)

with st.spinner('Model is being loaded..'):
  curr_model=load_model('AIvsREAL.h5')

st.write("AI generated Images vs Real Images")

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png","jpeg"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (126,126)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, curr_model)
    
    st.write(prediction)
    
    if(np.argmax(prediction)==1):
        print("This is a Real Image")
    else:
        print("This is an AI generated Image")
    
    
)
