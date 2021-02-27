import cv2
import dlib
import argparse

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='image file')
args = ap.parse_args()
# load input image
image = (args.image)


# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1('/content/darknet_PersonAndFaceDetection/mmod_human_face_detector.dat')
# apply face detection (cnn)
faces_cnn = cnn_face_detector(image, 1)
counting_people = 0
# loop over detected faces
for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y
     # draw box over face
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
    counting_people = counting_people + 1

caption = "{} {}".format("Total people in frame: ", str(counting_people))
cv2.putText(image, caption, (40,60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 100, 0), 2)
print(image)
