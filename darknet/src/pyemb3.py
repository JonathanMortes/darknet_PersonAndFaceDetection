import cv2
import dlib

def detectFaces(image):
    # initialize cnn based face detector with the weights
    cnn_face_detector = dlib.cnn_face_detection_model_v1("/content/darknet_PersonAndFaceDetection/mmod_human_face_detector.dat")
    # apply face detection (cnn)
    faces_cnn = cnn_face_detector(image, 1) #Number of pases
    counter = 0
    # loop over detected faces
    for face in faces_cnn:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        counter = counter + 1
         # draw box over face
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
    caption = "{} {}".format("Total faces in frame: ", str(counting_faces))
    print(caption)
    cv2.putText(image, caption, (40,100), cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 100), 2)
return image
