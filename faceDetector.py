#Face_detector
import numpy as np
import tensorflow as tf
import os
from keras import backend as K
from keras.layers import Input
from PIL import Image
import colorsys
import warnings
import cv2
import dlib


#code for facial detection
# apply face detection (cnn)
cnn_face_detector = dlib.cnn_face_detection_model_v1("/content/darknet_PersonAndFaceDetection/mmod_human_face_detector.dat")

input_video = cv2.VideoCapture("/content/darknet_PersonAndFaceDetection/people.avi")
output_video_filepath = "/content/darknet_PersonAndFaceDetection/detections.avi"
frame_width = int(input_video.get(3))
frame_height = int(input_video.get(4))
output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               20,
                               (frame_width, frame_height))
frame_count = 0
print("Starting facial detection....")
print("-----------------------------")
while (input_video.isOpened()):
    ret, frame = input_video.read()
    if (ret):
        print("Processing frame: ",frame_count)
        input_image = frame.copy()
        frame_count +=1
        image2detect = np.array(input_image, dtype=np.uint8)
        image2detect = cv2.cvtColor(image2detect, cv2.COLOR_BGR2RGB)
        faces_cnn = cnn_face_detector(image2detect, 1)
        # loop over detected faces
        counting_faces = 0
        for face in faces_cnn:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
            # draw box over face
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            counting_faces = counting_faces +1
        caption = "{} {}".format("Total faces in frame: ", str(counting_faces))
        print(caption)
        cv2.putText(frame, caption, (40,80), cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 100), 2)
        output_video.write(frame)
    else:
        print("Detection terminated. Saving as detections.avi'")
        break
input_video.release()
output_video.release()
