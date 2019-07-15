import numpy as np
import cv2

# find the triangle pts
# pts = [(100, 100), (100, 200), (400, 300)]

def crop_triangle(image, triangle_pts):

    # laod image
    img = cv2.imread(image)

    # find the triangle pts
    # pts = [(100, 100), (100, 200), (400, 300)]
    pts = triangle_pts
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

    return cropped_triangle, cropped_triangle_pts, rect

image = "images/bradley_cooper.jpg"
triangle_pts = [(100, 100), (100, 200), (400, 300)]
cropped_triangle, cropped_triangle_pts, rect = crop_triangle(image, triangle_pts)
cv2.imshow('cropped triangle', cropped_triangle)
cv2.waitKey(0)
cv2.destroyAllWindows()
