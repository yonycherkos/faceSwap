# import the necessary libraries
import numpy as np
import cv2
import dlib

# loading the images
img = cv2.imread('images/bradley_cooper.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('images/jim_carrey.jpg')
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)
img2_new = np.zeros_like(img2)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# face1
# part 1: landmarks of face1
faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

        # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(mask, convexhull, 255)
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    # part 2: Delaunay triangulation of face1
    rect = cv2.boundingRect(points)
    # (x, y, w, h) = rect
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    # print(triangles)
    triangles = np.array(triangles, np.int32)
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        # the index of the triangles
        index_pt1 = np.where((points == pt1).all(axis = 1))[0][0]
        index_pt2 = np.where((points == pt2).all(axis = 1))[0][0]
        index_pt3 = np.where((points == pt3).all(axis = 1))[0][0]
        indexes_triangles.append([index_pt1, index_pt2, index_pt3])

        # cropping triangles
        triangle_points = np.array([pt1, pt2, pt3], np.int32)
        rect = cv2.boundingRect(triangle_points)
        x, y, w, h = rect
        cropped_triangle = img[y :y + h, x :x + w]
        cropped_triangle_mask = np.zeros((h, w), np.uint8)
        cropped_triangle_points = np.array([[triangle_points[0][0] - x, triangle_points[0][1] - y],
                                            [triangle_points[1][0] - x, triangle_points[1][1] - y],
                                            [triangle_points[2][0] - x, triangle_points[2][1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_triangle_mask, cropped_triangle_points, (255, 255, 255))
        cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_triangle_mask)

        # drawing the triangles points
        # cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        # cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        # cv2.line(img, pt1, pt3, (0, 0, 255), 2)
# face2
# part1: landmarks of face2
faces = detector(img2_gray)
for face in faces:
    landmarks2 = predictor(img2_gray, face)
    landmarks_points2 = []
    for n in range(0, 68):
        x = landmarks2.part(n).x
        y = landmarks2.part(n).y
        landmarks_points2.append((x, y))

        # cv2.circle(img2, (x, y), 3, (0, 0, 255), -1)

# part2: triangulations of face2
for index in indexes_triangles:
    # tr2_pt1 = landmarks_points2[index[0]]
	# tr2_pt2 = landmarks_points2[index[1]]
	# tr2_pt3 = landmarks_points2[index[2]]
	# triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
    tr2_pt1 = landmarks_points2[index[0]]
    tr2_pt2 = landmarks_points2[index[1]]
    tr2_pt3 = landmarks_points2[index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2
    cropped_triangle2 = img2[y: y + h, x : x + w]
    cropped_triangle2_mask = np.zeros((h, w), np.uint8)
    cropped_triangle2_points = np.array([[tr2_pt1[0][0] - x, tr2_pt1[0][1] - y],
                                         [tr2_pt2[1][0] - x, tr2_pt2[1][1] - y],
                                         [tr2_pt3[2][0] - x, tr2_pt3[2][1] - y]])

    cv2.fillConvexPoly(cropped_triangle2_mask, cropped_triangle2_points, (255, 255, 255))
    cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask = cropped_triangle2_mask)


    # # drawing the triangles
    # cv2.line(img2, pt1, pt2, (0, 0, 255), 2)
    # cv2.line(img2, pt2, pt3, (0, 0, 255), 2)
    # cv2.line(img2, pt1, pt3, (0, 0, 255), 2)

    # warp triangle
    cropped_triangle_points = np.float32(cropped_triangle_points)
    cropped_triangle_points2 = np.float32(cropped_triangle2_points)
    M = cv2.getAffineTransform(cropped_triangle_points, cropped_triangle2_points)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))

    # reconstruct the second image
    img2_new[y :y + h, x : x + w] = warped_triangle
    break

# cv2.imshow('image 1', img)
# cv2.imshow('image 2', img2)
# cv2.imshow('mask', mask)
# cv2.imshow('face_image_1', face_image_1)
cv2.imshow('cropped_triangle', cropped_triangle)
cv2.imshow('cropped_triangle2', cropped_triangle2)
# cv2.imshow('cropped_triangle_mask', cropped_triangle_mask)
cv2.imshow('warped triangle', img2_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
