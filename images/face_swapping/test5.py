# import the necessary libraries
import numpy as np
import cv2
import dlib

def landmark_detection(image):

    # laod an image then convert it to grey scale
    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect the face then find the landmarks
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmark_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_points.append((int(x), int(y)))

            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)


    return landmark_points, img


image1 = 'ted_cruz.jpg'
image2 = 'donald_trump.jpg'

landmark_points1, landmarks_img1 = landmark_detection(image1)
landmark_points2, landmarks_img2 = landmark_detection(image2)

cv2.imshow("triangulation_img1", landmarks_img1)
cv2.imshow("triangulation_img2", landmarks_img2)

cv2.imwrite('images/landmarks_' + image1, landmarks_img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
