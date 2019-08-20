import numpy as np
import cv2
import dlib
from PIL import Image

# type of modes to apply face swapping
ALL_FACE_MODE = 'apply_on_all'
LARGEST_FACE_MODE = 'choose_largest_face'

# model file name for face landmark detection in dlib library
FACE_LANDMARK_SHAPE_DETECTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'

NUMBER_OF_FACE_LANDMARKS = 68   # number of facelandmark points


def shape_to_np_array(shape, dtype="int"):
    """
    Change shape data structure to np array
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((NUMBER_OF_FACE_LANDMARKS, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, NUMBER_OF_FACE_LANDMARKS):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def is_black_and_white(img):
    img_arr = np.array(img)
    channel_colors_match = (img_arr[:, :, 0] == img_arr[:, :, 1]) == (img_arr[:, :, 1] == img_arr[:, :, 2])
    if channel_colors_match.all() == True:
        return True
    return False


def match_image_color(src_img, dst_img):
    if is_black_and_white(src_img) is True and is_black_and_white(dst_img) is False:
        dst_img_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)
        dst_img_bw = cv2.cvtColor(dst_img_gray, cv2.COLOR_GRAY2BGR)
        dst_img = dst_img_bw
    elif is_black_and_white(src_img) is False and is_black_and_white(dst_img) is True:
        src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        src_img_bw = cv2.cvtColor(src_img_gray, cv2.COLOR_GRAY2BGR)
        src_img = src_img_bw
    else:
        pass
    return src_img, dst_img


def landmark_detection(img):
    """
    Generate facial landmark points of a give image.

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
    except Exception:
        raise ValueError("Image could not be converted to grayscale")

    # detect the face then find the landmarks points
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(
            FACE_LANDMARK_SHAPE_DETECTOR_FILENAME)
    except Exception:
        raise ValueError(
            "Facial landmark points file cann't be found. Download 'shape_predictor_68_face_landmarks.dat'")

    faces = detector(img_gray)
    if len(faces) == 0:
        raise ValueError('No face could be detected.')

    faces_landmark_points = []
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmark_points = shape_to_np_array(landmarks)
        faces_landmark_points.append(landmark_points)

    return faces_landmark_points


def choose_largest_face(faces_landmark_points):
    """
    Choose largest face from all the faces in a given image.

    Parameters
    ----------
    faces_landmark_points : list
        landmark points of all the faces.

    Returns
    -------
    largest_face_landmark_points: list
        return landmark points of the largest face.
    """
    largest_size = 0
    faces_len = len(faces_landmark_points)
    for face_i in range(faces_len):
        boundingRect = cv2.boundingRect(faces_landmark_points[face_i])

        face_size = boundingRect[2] * boundingRect[3]   # ? getting face size
        if face_size > largest_size:
            largest_size = face_size
            largest_face_index = face_i

    largest_face_landmark_points = faces_landmark_points[largest_face_index]

    return largest_face_landmark_points


def find_face_direction(landmark_points):
    """
    Find left or right direction of a face.

    Parameters
    ----------
    landmark_points : list
        Facial lanmark points of a given image.

    Returns
    -------
    direction: str
        Direction of the face.
    """

    # ? are these points enough
    pt1 = landmark_points[3]
    pt2 = landmark_points[34]
    pt3 = landmark_points[15]

    # ? are these criteria enough
    face_width = np.linalg.norm(np.subtract(pt3, pt1))
    left_dist = np.linalg.norm(np.subtract(pt2, pt1)) / face_width
    right_dist = np.linalg.norm(np.subtract(pt3, pt2)) / face_width

    # ? emprical constant 0.2
    if left_dist > right_dist + 0.2:
        direction = "right"
    elif right_dist > left_dist + 0.2:
        direction = "left"
    else:
        direction = "front"  # ? what happens when front

    return direction


def align_face_direction(src_img, src_landmark_points, dst_landmark_points):
    """
    Align the direction of the face of the two images.

    Parameters
    ----------
    src_img : nparray
        Numpy array of src_image.
    src_landmark_points : list
        Landmark points of src_image`.
    dst_landmark_points : list
        Landmark points of dst_image.

    Returns
    -------
    src_img: nparray
        The flipped or the original image numpy array.
    """

    src_direction = find_face_direction(src_landmark_points)
    dst_direction = find_face_direction(dst_landmark_points)

    # ? how to align if front face detected
    if (src_direction == "left" and dst_direction == "right") or (src_direction == "right" and dst_direction == "left"):
        flipped_src_img = cv2.flip(src_img, flipCode=1)
        return flipped_src_img

    return src_img


def applyConvexHull(src_landmark_points, dst_landmark_points):
    """
    Find the convex hull of each landmark points.

    Parameters
    ----------
    src_landmark_points : list
        a list of tuple integer of landmark points 1.
    dst_landmark_points : list
        a list of tuple integer of landmark points 2.

    Returns
    -------
    src_hull_points : list
        return a list of tuple integer of convex hull points bounding landmark points 1.
    dst_hull_points : list
        return a list of tuple integer of convex hull points bounding landmark points 2.
    """

    # Find convex hull of the two landmark points
    src_hull_points = []
    dst_hull_points = []

    src_hull_index = cv2.convexHull(src_landmark_points, returnPoints=False)
    dst_hull_index = cv2.convexHull(dst_landmark_points, returnPoints=False)

    # use the max number of convex hull points.
    # ? should we use larger or smaller(for the sake of only avoid index out of bound)
    if len(src_hull_index) > len(dst_hull_index):
        max_hull_index = src_hull_index
    else:
        max_hull_index = dst_hull_index

    for i in range(len(max_hull_index)):
        src_hull_points.append(src_landmark_points[int(max_hull_index[i])])
        dst_hull_points.append(dst_landmark_points[int(max_hull_index[i])])

    return src_hull_points, dst_hull_points


def calculateDelaunayTriangles(img, hull_points):
    """
    Calculate delauney triangles of a give points.

    Parameters
    ----------
    img : nparray
        image file path.
    hull_points : list
        hull points of the image.

    Returns
    -------
    delaunayTri_indexes : list
        return a list tuple integer contain the indexes of the landmark points.
    """
    # ? should we calculate all delanuay triangles
    (h, w, _) = img.shape
    rect = (0, 0, w, h)

    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(hull_points)
    triangleList = subdiv.getTriangleList()

    delaunayTri_indexes = []

    for t in triangleList:

        vtx1 = (t[0], t[1])
        vtx2 = (t[2], t[3])
        vtx3 = (t[4], t[5])
        vertexes = [vtx1, vtx2, vtx3]

        index = []  # ? this index
        # find the corresponding hull points of the triangleList
        # or find the indexes of the corresponding hull points
        for i in range(3):
            for j in range(0, len(hull_points)):
                if(vertexes[i][0] == hull_points[j][0] and vertexes[i][1] == hull_points[j][1]):
                    index.append(j)
                    break
        # Three points form a triangle
        if len(index) == 3:
            delaunayTri_indexes.append((index[0], index[1], index[2]))

    return delaunayTri_indexes


def applyAffineTransform(src, srcTri_offset, dstTri_offset, dst_size):
    """Warp src_image ROI using the convertion matrix.

    Parameters
    ----------
    src : numpy.ndarray
        src_image ROI which is to be warped.
    srcTri_offset : list
        single triangle points of src_image.
    dstTri_offset : list
        single triangle points of dst_image.
    dsize : tuple
        size(w, h) of dst_img ROI.

    Returns
    -------
    dst : numpy.ndarray
        warped src_image ROI.

    """
    # find convertion matrix from triangle1 to triangle2
    warpMat = cv2.getAffineTransform(
        np.float32(srcTri_offset), np.float32(dstTri_offset))

    # ? warping method
    (w, h) = dst_size
    dst = cv2.warpAffine(src, warpMat, (w, h), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


def warpTriangle(src_img, dst_img, srcTri, dstTri):
    """Warp srcTri to dstTri then replace triangle 2 portion of dst_img by triangle 1 portion of src_img.

    Parameters
    ----------
    src_img : numpy.ndarray
        output of src_image read by opencv.
    dst_img : numpy.ndarray
        output of dst_image read by opencv.
    srcTri : tuple
        a single triangle points of src_image.
    dstTri : tuple
        a single triangle points of dst_image.

    Returns
    -------
        does not return any value.
    """

    # Find bounding rectangle for each triangle
    srcTri_boundingRect = cv2.boundingRect(np.array(srcTri))
    dstTri_boundingRect = cv2.boundingRect(np.array(dstTri))

    # Offset points by left top corner of the respective rectangles
    srcTri_offset = []
    dstTri_offset = []

    for i in range(3):
        srcTri_offset.append(((srcTri[i][0] - srcTri_boundingRect[0]), (srcTri[i][1] - srcTri_boundingRect[1])))
        dstTri_offset.append(((dstTri[i][0] - dstTri_boundingRect[0]), (dstTri[i][1] - dstTri_boundingRect[1])))

    # Apply warpImage to small rectangular patches
    (src_x, src_y, src_w, src_h) = (srcTri_boundingRect[0], srcTri_boundingRect[1], srcTri_boundingRect[2], srcTri_boundingRect[3])
    src_img_rect_roi = src_img[src_y:src_y + src_h, src_x:src_x + src_w]

    (dst_x, dst_y, dst_w, dst_h) = (dstTri_boundingRect[0], dstTri_boundingRect[1], dstTri_boundingRect[2], dstTri_boundingRect[3])
    dst_img_rect_roi = applyAffineTransform(
        src_img_rect_roi, srcTri_offset, dstTri_offset, (dst_w, dst_h))

    # Get mask by filling triangle
    mask = np.zeros((dst_h, dst_w, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dstTri_offset), (1.0, 1.0, 1.0))
    dst_img_rect_roi = dst_img_rect_roi * mask

    # Copy triangular region of the rectangular patch to the output image
    dst_img[dst_y: dst_y + dst_h, dst_x: dst_x + dst_w] = dst_img[dst_y: dst_y + dst_h, dst_x: dst_x + dst_w] * ((1.0, 1.0, 1.0) - mask) + dst_img_rect_roi


def applyWarpTriangle(src_img, dst_img, dstTri_indexes, src_hull_points, dst_hull_points):
    """Compute warp triangles for each triangles of src_image and dst_image.first find.
        corresponding landmark points of each triangles from the triangulations
        indexes. then warp each triangle of src_image to dst_image by calling created
        warpTriangle function.

    Parameters
    ----------
    src_img : numpy.ndarray
        output of src_image read by opencv.
    dst_img : numpy.ndarray
        output of dst_image read by opencv.
    dstTri_indexes : list
        delaunay triangle indexes of dst_image.
    src_hull_points : list
        landmark points of src_image.
    dst_hull_points : list
        landmark points of dst_image.

    Returns
    -------
    dst_img : numpy.ndarray
        warped src_img copied to dst_img.

    """

    # iterate through each triangles
    for i in range(len(dstTri_indexes)):
        srcTri = []
        dstTri = []

        # iterate through all the three triangle indexes and find srcTri and dstTri
        for j in range(3):
            srcTri.append(src_hull_points[dstTri_indexes[i][j]])
            dstTri.append(dst_hull_points[dstTri_indexes[i][j]])

        warpTriangle(src_img, dst_img, srcTri, dstTri)

    return dst_img


def applySeamlessClone(dst_warped_img, dst_original_img, dstPoints):
    """Crop portion of dst_warped_img image and copy it to dst image.

    Parameters
    ----------
    dst_warped_img : nparray
        warped destination image.
    dst_original_img : nparray
        .
    dstPoints : type
        Description of parameter `dstPoints`.

    Returns
    -------
    swappedImage : numpy.nparray
        return portion dst_image replaced by portion of dst_warped_img_image.

    """

    # calculate mask
    mask = np.zeros(dst_original_img.shape, dtype=dst_original_img.dtype)
    cv2.fillConvexPoly(mask, np.int32(dstPoints), (255, 255, 255))

    # calculate center dst image where center of dst_warped_img image put
    rect = cv2.boundingRect(np.float32([dstPoints]))
    center = ((rect[0] + int(rect[2] / 2), rect[1] + int(rect[3] / 2)))

    # ? check MIXED_CLONE mode
    swappedImage = cv2.seamlessClone(
        dst_warped_img, dst_original_img, mask, center, cv2.NORMAL_CLONE)

    return swappedImage


def show_images(src_img, dst_img, swappedImage, showOriginalImages=False):
    """Display src_image, dst_image and warped image.

    Parameters
    ----------
    src_img : numpy.nparray
        output of src_image read with opencv.
    dst_img : numpy:nparray
        output of dst_image read with opencv.
    swappedImage : numpy.nparray
        the swapped image or new value of dst_image.

    Returns
    -------
    deosn't return any value. it just display the images.

    """

    if showOriginalImages:
        cv2.imshow("src_image", src_img)
        cv2.imshow("dst_image", dst_img)
    cv2.imshow("swappedImage", swappedImage)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def applyForBothModes(src_img, dst_img, dst_original_img, src_landmark_points, dst_landmark_points):
    """This function contain code tha will be use for both mode. inorder to avoid repetition.

    Parameters
    ----------
    src_img : nparray
        Readed value of src_image.
    dst_img : type
        Readed value of dst_image
    dst_original_img : type
        The original readed dst_img for backup.
    src_landmark_points : list
        List of facial landmark poinst of single face in src_image.
    dst_landmark_points : type
        List of facial landmark poinst of single face in dst_image.

    Returns
    -------
    swappedImage: numpy.ndarray
        The swapped image.

    """
    # find the convex hull bounding the landmark points of the images
    src_hull_points, dst_hull_points = applyConvexHull(
        src_landmark_points, dst_landmark_points)

    # calculate the delauney triangulations
    # srcTri_indexes = calculateDelaunayTriangles(
    #     src_img, src_hull_points)
    dstTi_indexes = calculateDelaunayTriangles(
        dst_img, dst_hull_points)

    dst_warped_img = applyWarpTriangle(
        src_img, dst_img, dstTi_indexes, src_hull_points, dst_hull_points)

    swappedImage = applySeamlessClone(
        dst_warped_img, dst_original_img, dst_hull_points)

    return swappedImage


def chooseModes(src_img, dst_img, dst_original_img, src_faces_landmark_points, dst_faces_landmark_points, mode="choose_largest_face"):
    """Choose ways to swap the images.

    Parameters
    ----------
    src_img : nparray
        Readed value of src_image.
    dst_img : type
        Readed value of dst_image
    dst_original_img : type
        The original readed dst_img for backup.
    src_faces_landmark_points : list
        landmark points of all the faces in src_image.
    dst_faces_landmark_points : list
        landmark points of all the faces in dst_image.
    mode : str
        Choose either 'choose_largest_face' or 'apply_on_all'.

    Returns
    -------
    swappedImage: numpy.ndarray
        The swapped image.

    """
    # check if there is only a single face in the second image
    if np.array(dst_faces_landmark_points).shape[0] == 1:
        mode = LARGEST_FACE_MODE

    if mode == ALL_FACE_MODE:
        faces = np.array(dst_faces_landmark_points).shape[0]
        for face in range(faces):
            src_landmark_points = src_faces_landmark_points[0]
            dst_landmark_points = dst_faces_landmark_points[face]

            swappedImage = applyForBothModes(
                src_img, dst_img, dst_original_img, src_landmark_points, dst_landmark_points)
            dst_original_img = swappedImage
    else:
        # find landmark points of the largest face in an image
        src_landmark_points = choose_largest_face(src_faces_landmark_points)
        dst_landmark_points = choose_largest_face(dst_faces_landmark_points)

        # match the direction of the two images
        src_img = align_face_direction(
            src_img, src_landmark_points, dst_landmark_points)

        # recompute lanmark points on the flipped image or new one.
        src_faces_landmark_points = landmark_detection(src_img)
        src_landmark_points = choose_largest_face(src_faces_landmark_points)

        swappedImage = applyForBothModes(
            src_img, dst_img, dst_original_img, src_landmark_points, dst_landmark_points)

    return swappedImage


def faceSwap(src_img, dst_img, mode="choose_largest_face", showImages=False):
    """Warping all up.
    Parameters
    ----------
    src_img: nparray
        Readed value of src_image.

    dst_img: nparray
        Readed value of dst_image.

    showOriginalImages : bool
        An optional argument whether or not to show the original images.

    Returns
    -------
    swappedImage: numpy.ndarray
        The swapped image.

    """
    # match the color of the two image
    src_img, dst_img = match_image_color(src_img, dst_img)

    # save the original dst_image
    dst_original_img = np.copy(dst_img)

    # find landmark points of the images
    src_faces_landmark_points = landmark_detection(src_img)
    dst_faces_landmark_points = landmark_detection(dst_img)

    swappedImage = chooseModes(
        src_img, dst_img, dst_original_img, src_faces_landmark_points, dst_faces_landmark_points, mode)

    if showImages:
        show_images(src_img, dst_original_img, swappedImage,
                    showOriginalImages=True)

    return swappedImage


if __name__ == '__main__':

    src_image = "faceswap/images/kalise.jpg"
    dst_image = "faceswap/images/black_and_white.jpg"

    src_img = cv2.imread(src_image)
    dst_img = cv2.imread(dst_image)

    swappedImage = faceSwap(src_img, dst_img, mode="choose_largest_face",
                            showImages=True)
