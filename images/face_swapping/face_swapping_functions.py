import numpy as np
import cv2
import dlib

def landmark_detection(image):

    # laod image
    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_pts = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            landmarks_pts.append((x, y))

    return landmarks_pts, img

def delaunay_triangulation(image, landmarks_pts):

    img = cv2.imread(image)

    pts = np.array(landmarks_pts, np.int32)
    rect = cv2.boundingRect(pts)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_pts)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, np.int32)
    pts = []
    for t in triangles:
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

def crop_triangle(image, pts):

    # laod image
    img = cv2.imread(image)

    # find the triangle pts
    # pts = [(100, 100), (100, 200), (400, 300)]
    pt1 = pts[0]
    pt2 = pts[1]
    pt3 = pts[2]
    pts = np.array(pts, np.int32)

    # draw the bounding box around the triangles
    rect = cv2.boundingRect(pts)
    (x, y, w, h) = rect
    cropped_triangle = img[y:y+h, x:x+w]

    # mask triangle
    cropped_triangle_mask = np.zeros((h, w), np.uint8)
    cropped_triangle_pts = np.array([[pt1[0] - x, pt1[1] - y],
                                     [pt2[0] - x, pt2[1] - y],
                                     [pt3[0] - x, pt3[1] - y]], np.int)
    cv2.fillConvexPoly(cropped_triangle_mask, cropped_triangle_pts, 255)
    cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_triangle_mask)

    size = [w, h]

    return cropped_triangle, cropped_triangle_pts, size

def warped_triangle(src_triangle, src_triangle_pts, dst_triangle_pts, dst_size):

    src_triangle_pts = np.float32(src_triangle_pts)
    dst_triangle_pts = np.float32(dst_triangle_pts)
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( src_triangle_pts, dst_triangle_pts )
    # Apply the Affine Transform just found to the src image
    warped_triangle = cv2.warpAffine( src_triangle, warpMat, (dst_size[0], dst_size[1]))

    return warped_triangle

def face_swapping(image1, image2, triangulation_pts1, triangulation_pts2):

    # wrapping all up
    # Read images
    src = cv2.imread("images/airplane.jpg")
    dst = cv2.imread("images/sky.jpg")

    # print("triangulation_pts1_len: ", len(triangulation_pts1)) = 112
    # print("triangulation_pts2_len: ", len(triangulation_pts2)) = 110
    # crop and warp every triangles of image1 into image2
    for t in range(len(triangulation_pts1)):
        cropped_triangle1, cropped_triangle1_pts, cropped_triangle_img1, size1 = crop_triangle(image1, triangulation_pts1[t])
        cropped_triangle2, cropped_triangle2_pts, cropped_triangle_img2, rect2 = crop_triangle(image2, triangulation_pts2[t])
        warped_triangle = warped_triangle(cropped_triangle1, cropped_triangle1_pts, cropped_triangle2_pts, size2)

        # Create a rough mask around the airplane.
        src_mask = np.zeros(src.shape, src.dtype)
        poly = np.array(triangulation_pts1[t], np.int32)
        cv2.fillPoly(src_mask, [poly], (255, 255, 255))

        # This is where the CENTER of the airplane will be placed
        center = (800,100)

        # Clone seamlessly.
        output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

        # Write result
        cv2.imwrite("images/opencv-seamless-cloning-example.jpg", output);

image1 = "images/bradley_cooper.jpg"
image2 = "images/jim_carrey.jpg"

img1 = cv2.imread(image1)
img2 = cv2.imread(image2)

landmarks_pts1, landmarks_img1 = landmark_detection(image1)
landmarks_pts2, landmarks_img2 = landmark_detection(image2)

triangulation_pts1, triangulation_img1 = delaunay_triangulation(image1, landmarks_pts1)
triangulation_pts2, triangulation_img2 = delaunay_triangulation(image2, landmarks_pts2)

# print("triangulation_pts1", triangulation_pts1[109])
# print("triangulation_pts2", triangulation_pts2[109])
#
# pt1_tri1, pt2_tri1, pt3_tri1 = triangulation_pts1[109][0], triangulation_pts1[109][1], triangulation_pts1[109][2]
# pt1_tri2, pt2_tri2, pt3_tri2 = triangulation_pts2[109][0], triangulation_pts2[109][1], triangulation_pts2[109][2]

# cv2.line(img1, pt1_tri1, pt2_tri1, (0, 0, 255), 2)
# cv2.line(img1, pt2_tri1, pt3_tri1, (0, 0, 255), 2)
# cv2.line(img1, pt1_tri1, pt3_tri1, (0, 0, 255), 2)
#
# cv2.line(img2, pt1_tri2, pt2_tri2, (0, 0, 255), 2)
# cv2.line(img2, pt2_tri2, pt3_tri2, (0, 0, 255), 2)
# cv2.line(img2, pt1_tri2, pt3_tri2, (0, 0, 255), 2)
#
# pts_tri1 = [(100, 100), (100, 200), (400, 300)]
# pts_tri2 = [(100, 100), (100, 200), (300, 300)]

cropped_triangle1, cropped_triangle1_pts, rect1 = crop_triangle(image1, triangulation_pts1[-1])
cropped_triangle2, cropped_triangle2_pts, rect2 = crop_triangle(image2, triangulation_pts2[-1])

# warped_triangle = warped_triangle(cropped_triangle1, cropped_triangle1_pts, cropped_triangle2_pts, rect2)

cv2.imshow('landmarks_img1', landmarks_img1)
cv2.imshow('landmarks_img2', landmarks_img2)
cv2.imshow('triangulation_img1', triangulation_img1)
cv2.imshow('triangulation_img2', triangulation_img2)
cv2.imshow('cropped_triangle1', cropped_triangle1)
cv2.imshow('cropped_triangle2', cropped_triangle2)
# cv2.imshow('warped_triangle', warped_triangle)

cv2.waitKey(0)
cv2.destroyAllWindows()
