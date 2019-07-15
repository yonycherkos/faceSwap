# import the necessary libraries
import numpy as np
import cv2
import dlib

# loading webcam
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
while True:
    ret, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)

    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(frame, face)
        landmarks_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            # cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        #cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)
        convex_image = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('webcam', frame)
    cv2.imshow('convex image', convex_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
