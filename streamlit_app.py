# -*- coding: utf-8 -*-
"""
@author: Babu Reddy
"""
#Import Libraries

from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np
import streamlit as st
import warnings
from typing import Union
warnings.simplefilter(action='ignore', category=FutureWarning)


 #ignore
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal 



face_classifier=cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

classifier=load_model(r'cnn_model.h5')

emotion_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

restore=st.empty()

from streamlit_webrtc import ClientSettings, WebRtcMode, webrtc_streamer

# setting for webcamera
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)


class Camera:
    '''
    Camera object to get video from remote source
    use read() method to read frames from video
    '''
    def __init__(self) -> None:
        self.webrtc_ctx = webrtc_streamer(key="loopback", mode=WebRtcMode.SENDONLY, client_settings=WEBRTC_CLIENT_SETTINGS)
    
    def read(self):
        if self.webrtc_ctx.video_receiver:
            try:
                frame = self.webrtc_ctx.video_receiver.get_frame()
            except queue.Empty:
                print("Queue is empty. Stop the loop.")
                self.webrtc_ctx.video_receiver.stop()
            img_rgb = frame.to_ndarray(format="rgb24")
            return img_rgb
        return None

cam = Camera()




while True:
   
    frame= cam.read()
    
    labels = []
    


    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    
    faces=face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) !=  0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position=(x,y-10)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
  
    restore.image(frame)
  
  
  
  



 

