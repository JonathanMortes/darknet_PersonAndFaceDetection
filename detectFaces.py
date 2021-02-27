import cv2
import dlib
import argparse

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='image file')
args = ap.parse_args()
view raw
# load input image
image = (args.image)


# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1('/content/darknet_PersonAndFaceDetection/mmod_human_face_detector.dat')
start = time.time()
# apply face detection (cnn)
faces_cnn = cnn_face_detector(image, 1)
counter = 0
# loop over detected faces
for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y
     # draw box over face
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
    counter = counter + 1
print('total faces detected')
print(counter)
