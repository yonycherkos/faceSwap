import numpy as np
import cv2
import dlib

# type of modes to apply face swapping
ALL_FACE_MODE = 'apply_on_all'
LARGEST_FACE_MODE = 'choose_largest_face'


class FaceSwap():
    """use to swap images"""

    def __init__(self):
        """Initilizing the class.

        Parameters
        ----------
        Doesn't take any Parameters

        Returns
        -------
            Doesn't return any value.

        """

        # self.image1 = image1
        # self.image2 = image2

    def landmark_detection(self, img):
        """Generate facial landmark points of a give image.

        Parameters
        ----------
        img: nparray
            Readed value of an image.

        Returns
        -------
        faces_landmark_points : list
            return landmark points for every face in a given image.

        """

        # convert the image to greyscaleprint("land here")
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise ValueError("Image could not be converted to grayscale")

        # detect the face then find the landmarks points
        detector = dlib.get_frontal_face_detector()
        try:
            predictor = dlib.shape_predictor(
                'shape_predictor_68_face_landmarks.dat')
        except Exception as e:
            raise ValueError(
                "facial landmark points file cann't be found. Download 'shape_predictor_68_face_landmarks.dat'")

        faces = detector(img_gray)
        if len(faces) == 0:
            raise ValueError('No face could be detected.')

        faces_landmark_points = []
        for face in faces:
            landmarks = predictor(img_gray, face)
            landmark_points = []
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmark_points.append((int(x), int(y)))

            faces_landmark_points.append(landmark_points)

        return faces_landmark_points

    def choose_largest_face(self, faces_landmark_points):
        """choose largest face from all the faces in a given image.

        Parameters
        ----------
        faces_landmark_points : list
            landmark points of all the faces.

        Returns
        -------
        largest_face_landmark_points: list
            return landmark points of the largest face.

        """

        size = 0
        faces = np.array(faces_landmark_points).shape[0]
        for face in range(faces):
            boundingRect = cv2.boundingRect(
                np.array(faces_landmark_points[face]))

            face_size = boundingRect[2] * boundingRect[3]
            if face_size > size:
                size = face_size
                larger_face_index = face

        largest_face_landmark_points = faces_landmark_points[larger_face_index]

        return largest_face_landmark_points

    # left or right face direction
    def find_face_direction(self, landmark_points):
        """Find left or right direction of a face.

        Parameters
        ----------
        landmark_points : list
            Facial lanmark points of a given image.

        Returns
        -------
        direction: str
            Direction of the face.

        """

        pt1 = landmark_points[3]
        pt2 = landmark_points[34]
        pt3 = landmark_points[15]

        face_width = np.linalg.norm(np.subtract(pt3, pt1))
        left_dist = np.linalg.norm(np.subtract(pt2, pt1)) / face_width
        right_dist = np.linalg.norm(np.subtract(pt3, pt2)) / face_width

        if left_dist > right_dist + 0.2:
            direction = "right"
        elif right_dist > left_dist + 0.2:
            direction = "left"
        else:
            direction = "front"

        return direction

    def alight_face_direction(self, img1, landmark_points1, landmark_points2):
        """Align the direction of the face of the two images.

        Parameters
        ----------
        img1 : nparray
            Numpy array of image1.
        landmark_points1 : list
            Landmark points of image1`.
        landmark_points2 : list
            Landmark points of image2.

        Returns
        -------
        img1: nparray
            The flipped or the original image numpy array.

        """

        img1direction = self.find_face_direction(landmark_points1)
        img2direction = self.find_face_direction(landmark_points2)

        if (img1direction == "left" and img2direction == "right") or (img1direction == "right" and img2direction != "left"):
            flipped_img1 = cv2.flip(img1, flipCode=1)
            img1 = flipped_img1
        else:
            img1 = img1

        return img1

    def applyConvexHull(self, landmark_points1, landmark_points2):
        """Find the convex hull of each landmark points.

        Parameters
        ----------
        points1 : list
            a list of tuple integer of landmark points 1.
        points2 : list
            a list of tuple integer of landmark points 2.

        Returns
        -------
        hull1 : list
            return a list of tuple integer of convex hull points bounding landmark points 1.
        hull2 : list
            return a list of tuple integer of convex hull points bounding landmark points 2.
        """

        # Find convex hull of the two landmark points
        hull1 = []
        hull2 = []

        hullIndex1 = cv2.convexHull(
            np.array(landmark_points1), returnPoints=False)
        hullIndex2 = cv2.convexHull(
            np.array(landmark_points2), returnPoints=False)

        # use the minimum number of convex hull points to avoid index out of bound.
        if len(hullIndex1) < len(hullIndex2):
            hullIndex = hullIndex1
        else:
            hullIndex = hullIndex2

        for i in range(len(hullIndex)):
            hull1.append(landmark_points1[int(hullIndex[i])])
            hull2.append(landmark_points2[int(hullIndex[i])])

        return hull1, hull2

    def calculateDelaunayTriangles(self, img, points):
        """Calculate delauney triangles of a give points.

        Parameters
        ----------
        image : str
            image file path.
        points : list
            landmark points of the image.

        Returns
        -------
        delaunayTri_indexes : list
            return a list tuple integer contain the indexes of the landmark points.

        """

        # img = cv2.imread(image)
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

            index = []
            # Get face-points (from 68 face detector) by coordinates
            for i in range(3):
                for j in range(0, len(points)):
                    if(pts[i][0] == points[j][0] and pts[i][1] == points[j][1]):
                        index.append(j)
            # Three points form a triangle
            if len(index) == 3:
                delaunayTri_indexes.append((index[0], index[1], index[2]))

        return delaunayTri_indexes

    def applyAffineTransform(self, src, srcTri, dstTri, dsize):
        """Warp image1 ROI using the convertion matrix.

        Parameters
        ----------
        src : numpy.ndarray
            image1 ROI which is to be warped.
        srcTri : list
            single triangle points of image1.
        dstTri : list
            single triangle points of image2.
        dsize : tuple
            size(w, h) of img2 ROI.

        Returns
        -------
        dst : numpy.ndarray
            warped image1 ROI.

        """
        # find convertion matrix from triangle1 to triangle2
        warpMat = cv2.getAffineTransform(
            np.float32(srcTri), np.float32(dstTri))

        dst = cv2.warpAffine(src, warpMat, (dsize[0], dsize[1]), None,
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        return dst

    def warpTriangle(self, img1, img2, t1, t2):
        """Warp t1 to t2 then replace triangle 2 portion of img2 by triangle 1 portion of img1.

        Parameters
        ----------
        img1 : numpy.ndarray
            output of image1 read by opencv.
        img2 : numpy.ndarray
            output of image2 read by opencv.
        t1 : tuple
            a single triangle points of image1.
        t2 : tuple
            a single triangle points of image2.

        Returns
        -------
            does not return any value.

        """

        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.array(t1))
        r2 = cv2.boundingRect(np.array(t2))

        # Offset points by left top corner of the respective rectangles
        t1_offset = []
        t2_offset = []

        for i in range(3):
            t1_offset.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2_offset.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        # Apply warpImage to small rectangular patches
        img1_roi = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        size = (r2[2], r2[3])  # size = (w, h) or (x, y)
        img2_roi = self.applyAffineTransform(
            img1_roi, t1_offset, t2_offset, size)

        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0))
        img2_roi = img2_roi * mask

        # Copy triangular region of the rectangular patch to the output image
        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3],
                                                              r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask) + img2_roi
        # img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

    def applyWarpTriangle(self, img1, img2, img2Tri, points1, points2):
        """Compute warp triangles for each triangles of image1 and image2.first find.
           corresponding landmark points of each triangles from the triangulations
           indexes. then warp each triangle of image1 to image2 by calling created
           warpTriangle function.

        Parameters
        ----------
        img1 : numpy.ndarray
            output of image1 read by opencv.
        img2 : numpy.ndarray
            output of image2 read by opencv.
        img2Tri : list
            delaunay triangle indexes of image2.
        points1 : list
            landmark points of image1.
        points2 : list
            landmark points of image2.

        Returns
        -------
        img2 : numpy.ndarray
            warped img1 copied to img2.

        """

        # iterate through each triangles
        for i in range(len(img2Tri)):
            t1 = []
            t2 = []

            # iterate through all the three triangle indexes and find t1 and t2
            for j in range(3):
                t1.append(points1[img2Tri[i][j]])
                t2.append(points2[img2Tri[i][j]])

            self.warpTriangle(img1, img2, t1, t2)

        return img2

    def applySeamlessClone(self, src, dst, dstPoints):
        """Crop portion of src image and copy it to dst image.

        Parameters
        ----------
        src : type
            Description of parameter `src`.
        dst : type
            Description of parameter `dst`.
        dstPoints : type
            Description of parameter `dstPoints`.

        Returns
        -------
        swappedImage : numpy.nparray
            return portion image2 replaced by portion of image1.

        """

        # calculate mask
        mask = np.zeros(dst.shape, dtype=dst.dtype)
        cv2.fillConvexPoly(mask, np.int32(dstPoints), (255, 255, 255))

        # calculate center dst image where center of src image put
        r = cv2.boundingRect(np.float32([dstPoints]))
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

        swappedImage = cv2.seamlessClone(
            src, dst, mask, center, cv2.NORMAL_CLONE)

        return swappedImage

    def showImages(self, img1, img2, swappedImage, showOriginalImages=False):
        """Display image1, image2 and warped image.

        Parameters
        ----------
        img1 : numpy.nparray
            output of image1 read with opencv.
        img2 : numpy:nparray
            output of image2 read with opencv.
        swappedImage : numpy.nparray
            the swapped image or new value of image2.

        Returns
        -------
        deosn't return any value. it just display the images.

        """

        if showOriginalImages:
            cv2.imshow("image1", img1)
            cv2.imshow("image2", img2)
        cv2.imshow("swappedImage", swappedImage)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def saveSwappedImage(self, swappedImage):
        """Save warped image to images/generated_images with image2 filename.

        Parameters
        ----------
        swappedImage : numpy.nparray
            Img2 after swapping.

        Returns
        -------
        return the warped image.

        """

        image2name = self.image2.split("/")[2]
        cv2.imwrite("images/generated_images/" + image2name, swappedImage)

        return swappedImage

    def applyForBothModes(self, img1, img2, img2_original, landmark_points1, landmark_points2):
        """This function contain code tha will be use for both mode. inorder to avoid repetition.

        Parameters
        ----------
        img1 : nparray
            Readed value of image1.
        img2 : type
            Readed value of image2
        img2_original : type
            The original readed img2 for backup.
        landmark_points1 : list
            List of facial landmark poinst of single face in image1.
        landmark_points2 : type
            List of facial landmark poinst of single face in image2.

        Returns
        -------
        swappedImage: numpy.ndarray
            The swapped image.

        """
        # find the convex hull bounding the landmark points of the images
        hull1, hull2 = self.applyConvexHull(
            landmark_points1, landmark_points2)

        # calculate the delauney triangulations
        # triangulation_indexes1 = self.calculateDelaunayTriangles(
        #     img1, hull1)
        triangulation_indexes2 = self.calculateDelaunayTriangles(
            img2, hull2)

        img2warped = self.applyWarpTriangle(
            img1, img2, triangulation_indexes2, hull1, hull2)

        swappedImage = self.applySeamlessClone(
            img2warped, img2_original, hull2)

        return swappedImage

    def chooseModes(self, img1, img2, img2_original, faces_landmark_points1, faces_landmark_points2, mode="choose_largest_face"):
        """Choose ways to swap the images.

        Parameters
        ----------
        img1 : nparray
            Readed value of image1.
        img2 : type
            Readed value of image2
        img2_original : type
            The original readed img2 for backup.
        faces_landmark_points1 : list
            landmark points of all the faces in image1.
        faces_landmark_points2 : list
            landmark points of all the faces in image2.
        mode : str
            Choose either 'choose_largest_face' or 'apply_on_all'.

        Returns
        -------
        swappedImage: numpy.ndarray
            The swapped image.

        """
        # if mode == "choose_largest_face":

        if mode == ALL_FACE_MODE:
            faces = np.array(faces_landmark_points2).shape[0]
            for face in range(faces):
                landmark_points1 = faces_landmark_points1[0]
                landmark_points2 = faces_landmark_points2[face]

                swappedImage = self.applyForBothModes(
                    img1, img2, img2_original, landmark_points1, landmark_points2)
                img2_original = swappedImage
        else:
            # find landmark points of the largest face in an image
            landmark_points1 = self.choose_largest_face(faces_landmark_points1)
            landmark_points2 = self.choose_largest_face(faces_landmark_points2)

            # match the direction of the two images
            img1 = self.alight_face_direction(
                img1, landmark_points1, landmark_points2)

            swappedImage = self.applyForBothModes(
                img1, img2, img2_original, landmark_points1, landmark_points2)

        return swappedImage

    def faceSwap(self, img1, img2, mode, showImages=False, saveSwappedImage=False):
        """Warping all up.

        Parameters
        ----------
        img1: nparray
            Readed value of image1.

        img2: nparray
            Readed value of image2.

        showOriginalImages : bool
            An optional argument whether or not to show the original images.

        saveSwappedImage : bool
            An optional argument whether or not to save the swapped image.

        Returns
        -------
        swappedImage: numpy.ndarray
            The swapped image.

        """

        # save the original image2
        img2_original = np.copy(img2)

        # find landmark points of the images
        faces_landmark_points1 = self.landmark_detection(img1)
        faces_landmark_points2 = self.landmark_detection(img2)

        swappedImage = self.chooseModes(
            img1, img2, img2_original, faces_landmark_points1, faces_landmark_points2, mode)

        if showImages is True:
            self.showImages(img1, img2_original, swappedImage,
                            showOriginalImages=True)

        if saveSwappedImage is True:
            self.saveSwappedImage(swappedImage)

        return swappedImage


if __name__ == '__main__':

    image1 = "images/original_images/yony.jpg"
    image2 = "images/original_images/anchorman.jpg"

    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    faceSwap = FaceSwap()
    swappedImage = faceSwap.faceSwap(img1, img2, mode="apply_on_all",
                                     showImages=True, saveSwappedImage=False)
