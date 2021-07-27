#Import Libraries

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import streamlit as st


 

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

classifier=load_model('cnn_model.h5')

emotion_labels=['Fear','Happy','Neutral','Sad','Surprise']

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

cam=Camera()

while True:
    frame= cam.read()
    labels = []
   

    
    #gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if crop.size==0:
     print("Ignored this - crop picture seems to be empty")
    else:
      gray=cv2.imwrite(frame, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    
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
