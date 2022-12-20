import xgboost as xgb
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from  PIL import Image, ImageEnhance

#Loading up the Classification model we created
#model = xgb.XGBClassifier()
#model.load_model('ph_model.json')

#from keras.initializers import glorot_uniform
#Reading the model from JSON file
#with open('ph_model.json', 'r') as json_file:
 #   json_savedModel= json_file.read()
    
#load the model architecture 
#model_j = tf.keras.models.model_from_json(json_savedModel)
#model_j.summary()

#model_j.load_weights('model_weights.h5')"""

#Caching the model for faster loading
@st.cache

image = Image.open(r'...\'ph_kit.jpg') #Brand logo image (optional)

st.title('pH Level Predictor')
st.image('ph_kit.jpg')
st.header('Enter the Color Image:')


#Create two columns with different width
col1, col2 = st.columns( [0.8, 0.2])
with col1:               # To display the header text using css style
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)
    
with col2:               # To display brand logo
    st.image(image,  width=150)
  
  #Add a header and expander in side bar
st.sidebar.markdown('<p class="font">My First Photo Converter App</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
     st.write("""
        Use this simple app to convert your favorite photo to a pencil sketch, a grayscale image or an image with blurring effect.  \n  \nThis app was created by Sharone Li as a side project to learn Streamlit and computer vision. Hope you enjoy!
     """)
  
uploaded_file = st.file_uploader("Choose a file", type=['jpg','png','jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
        st.image(image,width=300) 

   with col2:
        st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)
     
    # To convert to a string based IO:
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
   # string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)
