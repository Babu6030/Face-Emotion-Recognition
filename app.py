import streamlit as st
import numpy as np
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2



st.title("Facial Emotion Recognizer")

face_classifier=cv2.CascadeClassifier(r'https://github.com/Babu6030/Face-Emotion-Recognition/tree/main/Kaggle%20Notebooks/OpenCv%20Code')
classifier=load_model(r'https://github.com/Babu6030/Face-Emotion-Recognition/blob/main/cnn_model.h5')

emotion_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

cap=cv2.VideoCapture(0)
