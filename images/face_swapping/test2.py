import numpy as np
import cv2

# crop triangle
# laod images
img1 = cv2.imread('images/bradley_cooper.jpg')
img2 = cv2.imread('images/jim_carrey.jpg')

# triangle 1
# find points of triangle 1
pts_tri1 = [(100, 100), (100, 200), (400, 300)]
pt1_tri1 = pts_tri1[0]
pt2_tri1 = pts_tri1[1]
pt3_tri1 = pts_tri1[2]
pts_tri1 = np.array(pts_tri1, np.int32)

# bounding rectagle of triangle 1
rect1 = cv2.boundingRect(pts_tri1)
(x1, y1, w1, h1) = rect1
cropped_tri1 = img1[y1:y1+h1, x1:x1+w1]

# mask triangle 1
cropped_tri1_mask = np.zeros((h1, w1), np.uint8)
cropped_tri1_pts = np.array([[pt1_tri1[0] - x1, pt1_tri1[1] - y1],
                             [pt2_tri1[0] - x1, pt2_tri1[1] - y1],
                             [pt3_tri1[0] - x1, pt3_tri1[1] - y1]], np.int)
cv2.fillConvexPoly(cropped_tri1_mask, cropped_tri1_pts, 255)
cropped_tri1 = cv2.bitwise_and(cropped_tri1, cropped_tri1, mask=cropped_tri1_mask)


# triangle2
# find points of triangle 2
pts_tri2 = [(100, 100), (100, 200), (300, 300)]
pt1_tri2 = pts_tri2[0]
pt2_tri2 = pts_tri2[1]
pt3_tri2 = pts_tri2[2]
pts_tri2 = np.array(pts_tri2, np.int32)

# bounding rectagle of triangle 2
rect2= cv2.boundingRect(pts_tri2)
(x2, y2, w2, h2) = rect2
cropped_tri2= img2[y2:y2+h2, x2:x2+w2]

# mask triangle 2
cropped_tri2_mask = np.zeros((h2, w2), np.uint8)
cropped_tri2_pts = np.array([[pt2_tri2[0] - x2, pt1_tri1[1] - y2],
                             [pt2_tri2[0] - x2, pt2_tri2[1] - y2],
                             [pt3_tri2[0] - x2, pt3_tri2[1] - y2]], np.int32)
cv2.fillConvexPoly(cropped_tri2_mask, cropped_tri2_pts, 255)
cropped_tri2= cv2.bitwise_and(cropped_tri2, cropped_tri2, mask=cropped_tri2_mask)

# warp triangle
cropped_tri1_pts = np.float32(cropped_tri1_pts)
cropped_tri2_pts = np.float32(cropped_tri2_pts)
M = cv2.getAffineTransform(cropped_tri1_pts, cropped_tri2_pts)
warped_triangle = cv2.warpAffine(cropped_tri1, M, (w2, h2))

# face swap
src = img1
dst = img2

# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)
# poly = np.array([[100, 100], [100, 200], [400, 300]], np.int32)
poly = cropped_tri1_pts
cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# This is where the CENTER of the airplane will be placed
center = (500, 500)
swapped_triangle = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

# show the images
cv2.imshow('image 1', img1)
cv2.imshow('image 2', img2)
cv2.imshow('cropped triangle1', cropped_tri1)
cv2.imshow('cropped triangle2', cropped_tri2)
cv2.imshow('warped triangle', warped_triangle)
cv2.imshow('swapped triangle', swapped_triangle)
cv2.waitKey(0)
cv2.destroyAllWindows()
