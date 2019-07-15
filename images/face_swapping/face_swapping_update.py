# import the necessary libraries
import numpy as np
import cv2
import dlib

# landmarks detection
img = cv2.imread('images/bradley_cooper.jpg')
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
        landmarks_points.append((x, y))

        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    # delaunay triangulation
    h, w = img.shape[0], img.shape[1]
    # additional_points = [(0, 0), (w/2, 0), (w, 0), (0, h/2), (0, h), (w/2, h), (w, h), (w, h/2)]
    # landmarks_points.append(additional_points)
    points = np.array(landmarks_points, np.int32)
    rect = cv2.boundingRect(points)
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangle_lists = subdiv.getTriangleList()
    triangle_lists = np.array(triangle_lists, np.int32)

    for t in triangle_lists:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img, pt1, pt3, (0, 0, 255), 2)

        # cropping triangles of face1
        triangle_points = np.array([pt1, pt2, pt3], np.int32)
        rect = cv2.boundingRect(triangle_points)
        x, y, w, h = rect
        cropped_triangle = img[y :y + h, x :x + w]
        cropped_triangle_mask = np.zeros((h, w), np.uint8)
        cropped_triangle_points = np.array([[triangle_points[0][0] - x, triangle_points[0][1] - y],
                                            [triangle_points[1][0] - x, triangle_points[1][1] - y],
                                            [triangle_points[2][0] - x, triangle_points[2][1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_triangle_mask, cropped_triangle_points, 255)
        cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_triangle_mask)

        # drawing the triangles points
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img, pt1, pt3, (0, 0, 255), 2)

        # cropping the traingles of face2
        triangle_points2 = np.array([pt1, pt2, pt3], np.int32)
        rect2 = cv2.boundingRect(triangle_points2)
        x, y, w, h = rect2
        cropped_triangle2 = img2[y :y + h, x :x + w]
        cropped_triangle_mask2 = np.zeros((h, w), np.uint8)
        cropped_triangle_points2 = np.array([[triangle_points2[0][0], triangle_points2[0][1]],
                                            [triangle_points2[1][0], triangle_points2[1][1]],
                                            [triangle_points2[2][0], triangle_points2[2][1]]], np.int32)
        cv2.fillConvexPoly(cropped_triangle_mask2, cropped_triangle_points2, 255)
        cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_triangle_mask2)

        # # drawing the triangles
        # cv2.line(img2, pt1, pt2, (0, 0, 255), 2)
        # cv2.line(img2, pt2, pt3, (0, 0, 255), 2)
        # cv2.line(img2, pt1, pt3, (0, 0, 255), 2)

        # warp triangle
        cropped_triangle_points = np.float32(cropped_triangle_points)
        cropped_triangle_points2 = np.float32(cropped_triangle_points2)
        M = cv2.getAffineTransform(cropped_triangle_points, cropped_triangle_points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))

cv2.imshow('landmarks', img)
cv2.imshow('warped_triangle', warped_triangle)
cv2.waitKey(0)
cv2.destroyAllWindows()
