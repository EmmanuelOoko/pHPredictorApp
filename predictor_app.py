import xgboost as xgb
import streamlit as st
import pandas as pd

#Loading up the Classification model we created
model = xgb.XGBClassifier()
model.load_model('ph_model.json')

#Caching the model for faster loading
@st.cache
