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

def applyConvexHull(points1, points2):

    # Find convex hull of the two landmark points
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)

    for i in range(len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    return hull1, hull2


def calculateDelaunayTriangles(image, points):

    img = cv2.imread(image)
    rect = (0, 0, img.shape[1], img.shape[0])

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    triangleList = subdiv.getTriangleList()

    delaunayTri_indexes = []


    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        pts = [pt1, pt2, pt3]

        # draw the Triangulation
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img, pt1, pt3, (0, 0, 255), 2)

        index = []
        #Get face-points (from 68 face detector) by coordinates
        for i in range(3):
            for j in range(0, len(points)):
                if(pts[i][0] == points[j][0] and pts[i][1] == points[j][1]):
                    index.append(j)
        # Three points form a triangle
        if len(index) == 3:
            delaunayTri_indexes.append((index[0], index[1], index[2]))

    return delaunayTri_indexes, img

def applyAffineTransform(src, srcTri, dstTri, dsize) :

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (dsize[0], dsize[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.array(t1))
    r2 = cv2.boundingRect(np.array(t2))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []

    for i in range(3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])
    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2Rect), (1.0, 1.0, 1.0));
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask ) + img2Rect
    # img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def applyWarpTriangle(img1, img2, img2Tri, points1, points2):

    # Apply affine transformation to Delaunay triangles
    for i in range(len(img2Tri)):
        t1 = []
        t2 = []

        #get points for img1, img2 corresponding to the triangles
        for j in range(3):
            t1.append(points1[img2Tri[i][j]])
            t2.append(points2[img2Tri[i][j]])

        warpTriangle(img1, img2, t1, t2)

    return img2


def applySeamlessClone(src, dst, dstPoints):

    # calculate mask
    mask = np.zeros(dst.shape, dtype = dst.dtype)
    cv2.fillConvexPoly(mask, np.int32(dstPoints), (255, 255, 255))

    # calculate center dst image where center of src image put
    r = cv2.boundingRect(np.float32([dstPoints]))
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

    warpedImage = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)

    return warpedImage


def approachMode(approach, hull1, hull2, landmark_points1, landmark_points2):

    # use the two approachs
    if approach == "approach1":
        points1 = hull1
        points2 = hull2
    else:
        points1 = landmark_points1
        points2 = landmark_points2

    return points1, points2


def showImages(img1, img2, warpedImage):

    cv2.imshow("image1", img1)
    cv2.imshow("image2", img2)
    cv2.imshow("warpedImage", warpedImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def saveSwappedImage(warpedImage, image2, approach):

    image2name = image2.split("/")[2]
    cv2.imwrite("images/generated_images/" + approach  + "/"+ image2name, warpedImage)


image1 = 'images/original_images/ted_cruz.jpg'
image2 = 'images/original_images/26-Nba-memes-13.jpg'

img1 = cv2.imread(image1)
img2 = cv2.imread(image2)
img2_original = np.copy(img2)

landmark_points1, landmarks_img1 = landmark_detection(image1)
landmark_points2, landmarks_img2 = landmark_detection(image2)

hull1, hull2 = applyConvexHull(landmark_points1, landmark_points2)

approach = "approach1"
points1, points2 = approachMode(approach, hull1, hull2, landmark_points1, landmark_points2)

triangulation_indexes1, triangulation_img1 = calculateDelaunayTriangles(image1, points1)
triangulation_indexes2, triangulation_img2 = calculateDelaunayTriangles(image2, points2)

img2warped = applyWarpTriangle(img1, img2, triangulation_indexes2, points1, points2)

warpedImage = applySeamlessClone(img2warped, img2_original, points2)


showImages(img1, img2_original, warpedImage)

saveSwappedImage(warpedImage, image2, approach)
