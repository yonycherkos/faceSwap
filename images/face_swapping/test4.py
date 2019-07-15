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


def calculateDelaunayTriangles1(image, points):

    img = cv2.imread(image)
    rect = (0, 0, img.shape[1], img.shape[0])

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    triangleList = subdiv.getTriangleList()

    pts = []
    delaunayTri = []

    for t in triangleList:
        # extract the triangle pts
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        pts.append([pt1, pt2, pt3])

        # draw the Triangulation
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img, pt1, pt3, (0, 0, 255), 2)

    return pts, img


def calculateDelaunayTriangles2(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv
    subdiv.insert(points)

    triangleList = subdiv.getTriangleList();

    delaunayTri = []


    for t in triangleList:

        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])


        ind = []
        #Get face-points (from 68 face detector) by coordinates
        for i in range(0, 3):
            for j in range(0, len(points)):
                if(pt[i][0] == points[j][0] and pt[i][1] == points[j][1]):
                    ind.append(j)
        # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
        if len(ind) == 3:
            delaunayTri.append((ind[0], ind[1], ind[2]))


    return delaunayTri

# laod imag"
image = "images/jim_carrey.jpg"
img = cv2.imread(image)

landmark_points,_ = landmark_detection(image)
rect = (0, 0, img.shape[1], img.shape[0])

pts, img = calculateDelaunayTriangles1(image, landmark_points)
delaunayTri = calculateDelaunayTriangles2(rect, landmark_points)


print("calculateDelaunayTriangles1   ", len(pts), "\n", pts)
print("calculateDelaunayTriangles2   ", len(delaunayTri), "\n", delaunayTri)

for t in
