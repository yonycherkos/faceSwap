# include the necessary libraries
import numpy as np
import cv2
import dlib

class FaceSwapping():
    """docstring for ."""

    def __init__(self):
        pass

    def landmark_detection(image):

        img = cv2.imread(image)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        faces = detector(img_gray)
        for face in faces:
            landmarks = predictor(img_gray, face)
            landmarks_points = []
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                landmarks_points.append((x, y))
        cv2.imshow('landmarks', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return landmarks_points


    def delaunay_triangulation(image, landmarks_points):

        # loading the images
        img = cv2.imread(image)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detector = dlib.get_frontal_face_detector()
        faces = detector(img_gray)
        for face in faces:
            points = np.array(landmarks_points, np.int32)
            rect = cv2.boundingRect(points)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(landmarks_points)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, np.int32)
            indexes_triangles = []
            pts = []
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                cv2.line(img, pt1, pt2, (0, 0, 255), 2)
                cv2.line(img, pt2, pt3, (0, 0, 255), 2)
                cv2.line(img, pt1, pt3, (0, 0, 255), 2)

                pts.append([pt1, pt2, pt3])

        cv2.imshow('triangulations', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return pts

    def face_matching(arg):
        pass

    def face_swapping(arg):
        pass

face_swapping = FaceSwapping()
image = "images/bradley_cooper.jpg"
landmarks_points = face_swapping.landmark_detection(image)
pts = delaunay_triangulation(image, landmarks_points)
# face_swapping(img1, img2)
