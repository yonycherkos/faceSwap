# import the necessary libraries
import numpy as np
import cv2
import dlib

def landmarks_to_txt(image):

    # laod image
    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        fd = open(image + ".txt", "w")
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            fd.write(str(x) + " " + str(y) + "\n")

        fd.close()

    return 0

image1 = "bradley_cooper.jpg"
image2 = "jim_carrey.jpg"
landmarks_to_txt(image1)
landmarks_to_txt(image2)
