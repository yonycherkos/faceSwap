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


def calculateDelaunayTriangles(image, points):

    img = cv2.imread(image)
    rect = (0, 0, img.shape[1], img.shape[0])

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    triangleList = subdiv.getTriangleList()

    delaunayTri_indexes = []


    for t in triangleList:

        pts = []
        pts.append((t[0], t[1]))
        pts.append((t[2], t[3]))
        pts.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # draw the Triangulation
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img, pt1, pt3, (0, 0, 255), 2)

        index = []
        #Get face-points (from 68 face detector) by coordinates
        for i in range(0, 3):
            for j in range(0, len(points)):
                if(pts[i][0] == points[j][0] and pts[i][1] == points[j][1]):
                    index.append(j)
        # Three points form a triangle
        if len(index) == 3:
            delaunayTri_indexes.append((index[0], index[1], index[2]))

    return delaunayTri_indexes, img

def applyAffineTransform(src, srcTri, dstTri, size) :

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []

    for i in range(3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0));

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask ) + img2Rect
    # img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect


image1 = 'images/ted_cruz.jpg'
image2 = 'images/donald_trump.jpg'

img1 = cv2.imread(image1)
img2 = cv2.imread(image2)
img1Warped = np.array(img2)

landmark_points1, landmarks_img1 = landmark_detection(image1)
landmark_points2, landmarks_img2 = landmark_detection(image2)

# Find convex hull
hull1 = []
hull2 = []

hullIndex = cv2.convexHull(np.array(landmark_points2), returnPoints = False)

for i in range(len(hullIndex)):
    hull1.append(landmark_points1[int(hullIndex[i])])
    hull2.append(landmark_points2[int(hullIndex[i])])

triangulation_indexes1, triangulation_img1 = calculateDelaunayTriangles(image1, hull1)
triangulation_indexes2, triangulation_img2 = calculateDelaunayTriangles(image2, hull2)

dt = triangulation_indexes2
# Apply affine transformation to Delaunay triangles
for i in range(len(dt)):
    t1 = []
    t2 = []

    #get points for img1, img2 corresponding to the triangles
    for j in range(3):
        t1.append(hull1[dt[i][j]])
        t2.append(hull2[dt[i][j]])

    warpTriangle(img1, img1Warped, t1, t2)

# seamless cloning
# src and dst
src = img1Warped
dst = img2

# calculate mask
mask = np.zeros(img2.shape, dtype = img2.dtype)
cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))

# calculate center dst image where center of src image put
r = cv2.boundingRect(np.float32([hull2]))
center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

warpedImage = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)

cv2.imshow("triangulation_img1", img1)
cv2.imshow("triangulation_img2", img2)
cv2.imshow("warpedImage", warpedImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
