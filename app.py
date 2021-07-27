import streamlit as st
import numpy as np
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2



st.title("Facial Emotion Recognizer")

face_classifier=cv2.CascadeClassifier('/Babu6030/Face-Emotion-Recognition/tree/main/Kaggle%20Notebooks/OpenCv%20Code')
classifier=load_model('/Babu6030/Face-Emotion-Recognition/blob/main/Kaggle%20Notebooks/OpenCv%20Code/modelbestweights.h5')

emotion_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

cap=cv2.VideoCapture(0)
