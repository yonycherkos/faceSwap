import numpy as np
import cv2
import dlib
from PIL import Image
import grpc

# type of modes to apply face swapping
ALL_FACE_MODE = 'apply_on_all'
LARGEST_FACE_MODE = 'choose_largest_face'

# type of image arguments for referring image in error messages
ARG_IMG_SRC = 'face'
ARG_IMG_DST = 'meme'

# model file name for face landmark detection in dlib library
FACE_LANDMARK_SHAPE_DETECTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'

NUMBER_OF_FACE_LANDMARKS = 68   # number of facelandmark points


class GRPCException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return "GRPC Exception {}: {}".format(self.code, self.message)


def shape_to_np_array(shape, points_num=NUMBER_OF_FACE_LANDMARKS, dtype="int32"):
    """
    Change shape data structure to np array

    Args:
        shape: dlib shape data type
        dtype: numpy data type

    Returns:
        np_array from dlib shape data type
    """
    # initialize our nparray
    np_array = np.zeros((points_num, 2), dtype=dtype)

    # loop over all points and convert them to (x, y) tuples
    for i in range(points_num):
        np_array[i] = (shape.part(i).x, shape.part(i).y)

    return np_array


def is_black_and_white(img):
    img_arr = np.array(img)
    channel_colors_match = (img_arr[:, :, 0] == img_arr[:, :, 1]) == (
        img_arr[:, :, 1] == img_arr[:, :, 2])
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


def preprocess_images(src_img, dst_img):
    """
    All required preprocess for both images

    Args: 
        src_img: cvimage
        dst_img: cvimage

    Returns:
        cvimage: preprocessed src_img
        cvimage: preprocessed dst_img
    """
    src_img, dst_img = match_image_color(src_img, dst_img)

    return src_img, dst_img


def get_face_landmark_points(img, arg_img_type, largest_only=False):
    """
    Generate facial landmark points of a given image

    Args:
        img: nparray
            Readed value of an image
        arg_image_type: str from { 'face', 'meme' }
            wherther img is input face image or meme image for error message purposes
        largest_only: boolean
            return the largest face only

    Returns:
        list of nparrays or an nparray:
            list of landmark points for every face in a given image
            or a single landmark points for largest face if largest only
    """

    # convert the image to greyscaleprint("land here")
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        raise GRPCException(
            "{} image could not be converted to grayscale. "
            "Please, check if appropriate image is selected".format(arg_img_type))

    # detect the face then find the landmarks points
    detector = dlib.get_frontal_face_detector()
    try:
        predictor = dlib.shape_predictor(
            FACE_LANDMARK_SHAPE_DETECTOR_FILENAME)
    except Exception:
        raise GRPCException(grpc.StatusCode.FAILED_PRECONDITION,
                            "Facial landmark points file cann't be found. "
                            "Download 'shape_predictor_68_face_landmarks.dat'")

    faces = detector(img_gray)
    if len(faces) == 0:
        raise GRPCException(grpc.StatusCode.FAILED_PRECONDITION,
                            'No face could be detected from {} image input.'.format(arg_img_type))

    faces_landmark_points = []
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmark_points = shape_to_np_array(landmarks)
        faces_landmark_points.append(landmark_points)

    if largest_only:
        faces_landmark_points = choose_largest_face(faces_landmark_points)
    elif len(faces_landmark_points) > 1:
        faces_landmark_points = fix_landmark_overlap(faces_landmark_points)

    return faces_landmark_points


def rectContains(rect, point):
    """Check if a point is inside a rectangle.

    Parameters
    ----------
    rect : tuple
        Points of the rectangle edges.
    point : tuple
        List of points.

    Returns
    -------
    bool
        Return true if the points are inside rectangle else return false.

    """
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


def is_overlap(Rect1, Rect2):
    """Check if two rectangle are overlapped.

    Parameters
    ----------
    Rect1 : tuple
        Points of the rectangle1 edges.
    Rect2 : tuple
        Points of the rectangle2 edges.

    Returns
    -------
    bool
        Return true is overlap else return false.

    """
    if rectContains(Rect1, Rect2) or rectContains(Rect2, Rect1):
        return True
    else:
        return False


def fix_landmark_overlap(faces_landmark_points):
    """Disregared the overlapped faces(one of them).

    Parameters
    ----------
    faces_landmark_points : list
        Landmark points of each faces.

    Returns
    -------
    bool
        return the non overlapped faces landmark points.

    """
    none_overlap_faces_landmark_points = []
    faces = len(faces_landmark_points)
    overlapped_faces = []  # store the overlapped faces
    for i in range(faces):
        for j in range(faces):
            # check if face[i] and face[j] are the same face and are not overlapped with any other faces
            if (i == j) and (j not in overlapped_faces):
                none_overlap_faces_landmark_points.append(
                    faces_landmark_points[i])
            # check if the bounding box are overlapped
            elif is_overlap(cv2.boundingRect(faces_landmark_points[i]), cv2.boundingRect(faces_landmark_points[j])):
                overlapped_faces.append(j)
                continue
    return none_overlap_faces_landmark_points


def choose_largest_face(faces_landmark_points):
    """
    Choose largest face from all the faces in a given image

    Args:
        faces_landmark_points : list of nparrays
            landmark points of all the faces

    Returns:
        nparray:
            landmark points of the largest face
    """
    faces_num = len(faces_landmark_points)

    # if only one face return it
    if faces_num == 1:
        return faces_landmark_points[0]

    # search for the largest face otherwise
    largest_size = 0
    for face_i in range(faces_num):
        x, y, w, h = cv2.boundingRect(faces_landmark_points[face_i])

        face_size = w * h   # ? getting face size
        if face_size > largest_size:
            largest_size = face_size
            largest_face_index = face_i

    largest_face_landmark_points = faces_landmark_points[largest_face_index]

    return largest_face_landmark_points


def find_face_direction(landmark_points):
    """
    Find left or right direction of a face.

    Args:
        landmark_points : nparray
            Facial lanmark points of a given image

    Returns:
        direction: str
            Direction of the face
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
    Flip src_img based on the alignment of two faces

    Args:
        src_img : nparray
            Numpy array of src_img.
        src_landmark_points : nparray
            Landmark points of a face from src_img
        dst_landmark_points : nparray
            Landmark points of a face from dst_img

    Returns:
        nparray:
            The flipped or the original image
    """

    src_direction = find_face_direction(src_landmark_points)
    dst_direction = find_face_direction(dst_landmark_points)

    # ? how to align if front face detected
    if (src_direction == "left" and dst_direction == "right") or (src_direction == "right" and dst_direction == "left"):
        flipped_src_img = cv2.flip(src_img, flipCode=1)
        return flipped_src_img

    return src_img


def get_convex_hulls(src_face_points, dst_face_points):
    """
    Find the convex hulls for both landmark points

    Args:
        src_face_points : nparray
            face landmark points from src_image
        dst_face_points : nparray
            face landmark points from dst_image

    Returns:
        list:
            convex hull points bounding src face points
        list:
            convex hull points bounding dst face points
    """

    # Find convex hull of the two landmark points
    src_hull_points = []
    dst_hull_points = []

    src_hull_indexes = cv2.convexHull(src_face_points, returnPoints=False)
    dst_hull_indexes = cv2.convexHull(dst_face_points, returnPoints=False)

    # use the max number of convex hull points.
    # ? should we use larger or smaller(for the sake of only avoid index out of bound)
    if len(src_hull_indexes) > len(dst_hull_indexes):
        max_hull_indexes = src_hull_indexes
    else:
        max_hull_indexes = dst_hull_indexes

    for i in range(len(max_hull_indexes)):
        src_hull_points.append(src_face_points[int(max_hull_indexes[i])])
        dst_hull_points.append(dst_face_points[int(max_hull_indexes[i])])

    return src_hull_points, dst_hull_points


def get_corresponding_delaunays(src_points, dst_points, use_dst=True):
    """
    Calculate corresponding delauney triangles of a src and dst points

    Args:
        src_points : list of nparray
        dst_points : list of nparray   
        use_dst : if to use dst_points for triangle calculation

    Returns:
        tuple:
            0: src delauney triangles list
            1: dst delauney triangles list
    """
    # ? should we calculate all delanuay triangles
    # which points use calculating delanauy triangles
    points = dst_points if use_dst else src_points
    points_rect = cv2.boundingRect(np.array(points))

    subdiv = cv2.Subdiv2D(points_rect)
    subdiv.insert(points)
    triangleList = subdiv.getTriangleList()

    src_delaunay_tris = []
    dst_delaunay_tris = []

    for t in triangleList:
        vtx1 = (t[0], t[1])
        vtx2 = (t[2], t[3])
        vtx3 = (t[4], t[5])
        vertexes = [vtx1, vtx2, vtx3]

        src_triangle = []
        dst_triangle = []
        # Get face-points (from 68 face detector) by coordinates
        for i in range(3):
            for j in range(0, len(points)):
                if(vertexes[i][0] == points[j][0] and vertexes[i][1] == points[j][1]):
                    src_triangle.append(src_points[j])
                    dst_triangle.append(dst_points[j])
                    break
        # Three points form a triangle
        assert(len(src_triangle) == 3 and len(dst_triangle) == 3)
        src_delaunay_tris.append(src_triangle)
        dst_delaunay_tris.append(dst_triangle)

    return src_delaunay_tris, dst_delaunay_tris


def extract_face(img, face_points):
    """
    Crop image based on face_points to extract face 
    and transform face points corresponding to the cropped image

    Args:
        img: cvimage
        face_points: landmark points of face in img

    Returns:
        tuple:
            0: cropped cvimage
            1: cropped image box in [x,y,w,h] format
    """
    x, y, w, h = cv2.boundingRect(face_points)

    face_points[:, 0] -= x
    face_points[:, 1] -= y

    cropped_img = img[y:y+h, x:x+w]

    return cropped_img, [x, y, w, h]


def put_image(big_img, small_img, box):
    """
    Puts a small image on a big image at specified box

    Args:
        big_img: cvimage
        small_img: cvimage
        box: where to put image in [x,y,w,h] format
             w and h must be the same with small_img width and height

    Returns:
        changed cvimage
    """
    x, y, w, h = box
    assert(h == small_img.shape[0] and w == small_img.shape[1])
    big_img[y:y+h, x:x+w] = small_img

    return big_img


def replace_triangle(src_img, dst_img, src_tri_points, dst_tri_points, DEBUG=False):
    """
    Warp src_tri to dst_tri;then replace dst_tri portion of dst_img by src_tri portion of src_img
    Changes dst_img by replacing the warped triangle

    Args:
        src_img: numpy.ndarray
            src_img
        dst_img : numpy.ndarray
            dst_img
        src_tri_points : list of triple points
            a single triangle points from src_img
        dst_tri_points : list of triple points
            a single triangle points of dst_img
    """

    # Find bounding rectangle for each triangle
    src_tri_roi = cv2.boundingRect(np.array(src_tri_points))
    src_roi_x, src_roi_y, src_roi_w, src_roi_h = src_tri_roi

    dst_tri_roi = cv2.boundingRect(np.array(dst_tri_points))
    dst_roi_x, dst_roi_y, dst_roi_w, dst_roi_h = dst_tri_roi

    # Transform points by left top corner of the respective rectangles
    transformed_src_tri = []
    transformed_dst_tri = []

    if DEBUG:
        dst_tri_vtxs = np.array(dst_tri_points)
        dst_roi_vtxs = np.array([
            (dst_roi_x, dst_roi_y),
            (dst_roi_x+dst_roi_w, dst_roi_y),
            (dst_roi_x+dst_roi_w, dst_roi_y+dst_roi_h),
            (dst_roi_x, dst_roi_y+dst_roi_h)
        ])
        cv2.drawContours(dst_img, [dst_tri_vtxs], 0, (255, 0, 0), 1)
        # cv2.drawContours(dst_img, [rect_vtxs], 0, (0, 0, 255), 1)
    for vtx_i in range(3):
        if DEBUG:
            cv2.circle(dst_img, tuple(dst_tri_points[vtx_i]), 2, (255, 0, 0))

        transformed_src_tri.append(
            ((src_tri_points[vtx_i][0] - src_roi_x), (src_tri_points[vtx_i][1] - src_roi_y)))
        transformed_dst_tri.append(
            ((dst_tri_points[vtx_i][0] - dst_roi_x), (dst_tri_points[vtx_i][1] - dst_roi_y)))

    # Apply warpImage to small rectangular patches
    src_img_roi = src_img[src_roi_y:src_roi_y +
                          src_roi_h, src_roi_x:src_roi_x + src_roi_w]

    # find convertion matrix from src_tri to dst_tri
    warpMat = cv2.getAffineTransform(
        np.float32(transformed_src_tri), np.float32(transformed_dst_tri))

    # ? warping method
    dst_roi_dims = (dst_roi_w, dst_roi_h)
    warped_src_img_roi = cv2.warpAffine(src_img_roi, warpMat, dst_roi_dims, None,
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Get mask by filling triangle
    warped_src_mask = np.zeros(
        warped_src_img_roi.shape, warped_src_img_roi.dtype)
    cv2.fillConvexPoly(warped_src_mask, np.int32(
        transformed_dst_tri), (1, 1, 1))
    warped_src_img_roi = warped_src_img_roi * warped_src_mask

    # Copy triangular region of the rectangular patch to the output image
    dst_img_roi = dst_img[dst_roi_y:dst_roi_y +
                          dst_roi_h, dst_roi_x:dst_roi_x + dst_roi_w]
    dst_roi_mask = (1, 1, 1) - warped_src_mask
    dst_img[dst_roi_y:dst_roi_y + dst_roi_h, dst_roi_x:dst_roi_x + dst_roi_w] \
        = dst_img_roi * dst_roi_mask + warped_src_img_roi

    if DEBUG:
        cv2.imshow('warped_src_img_roi', warped_src_img_roi)
        cv2.imshow('warped_src_mask', warped_src_mask)
        cv2.imshow('dst_img_roi', dst_img_roi)
        cv2.imshow('dst_img', dst_img)
        cv2.waitKey()


def replace_all_triangles(src_img, dst_img, src_tris, dst_tris):
    """
    Compute warp triangles for each triangles of image1 and image2.first find.
    corresponding landmark points of each triangles from the triangulations
    indexes. then warp each triangle of image1 to image2 by calling created
    warpTriangle function.
    Changes dst_img by replacing all delaunay triangles

    Args:
        src_img: numpy.ndarray
        dst_img: numpy.ndarray
        src_tris : list
            triangles for src image
        dst_tris : list
            triangles for dst image
    """

    # iterate through each triangles
    for triangle_i in range(len(src_tris)):
        replace_triangle(
            src_img, dst_img, src_tris[triangle_i],  dst_tris[triangle_i])


def apply_seamless_clone(src_img, dst_img, src_points, DEBUG=False):
    """
    Seamlessly clone portion of src image and copy it to dst image

    Args:
        src_img : cvimage
        dst_img : cvimage
        src_points : list of ndarray
            points used to creat mask where src_img will be placed on dst_img

    Returns:
        numpy.nparray
            dst_img replaced by portion of src_img
    """
    assert(src_img.shape == dst_img.shape)
    # calculate center dst image where center of src image put
    h = dst_img.shape[0]
    w = dst_img.shape[1]
    dst_points_center = ((round(w / 2), round(h / 2)))

    # calculate mask
    mask = np.zeros(src_img.shape, dtype=dst_img.dtype)
    cv2.fillConvexPoly(mask, np.int32(src_points), (255, 255, 255))

    # ? check MIXED_CLONE mode
    swappedImage = cv2.seamlessClone(
        src_img, dst_img, mask, dst_points_center, cv2.NORMAL_CLONE)
    if DEBUG:
        cv2.imshow('swap', swappedImage)
        cv2.waitKey(0)

    return swappedImage


def swap_using_points(src_img, dst_img, src_points, dst_points):
    """
    This function contain code tha will be use for both mode. inorder to avoid repetition

    Args:
        src_img : nparray
            Readed value of source image
        dst_img : nparray
            Readed value of dst_img
        processed_dst_img : nparray
            Copy of dst image that may be changed by previous swaps
        src_face_points : nparray
            List of facial landmark poinst of single face from source image
        dst_face_points : nparray
            List of facial landmark poinst of single face from destination image

    Returns:
        numpy.ndarray
            The swapped image
    """

    # calculate the delauney triangulations
    src_delaunays, dst_delaunays = get_corresponding_delaunays(
        src_points, dst_points, use_dst=True)

    dst_img_copy = np.copy(dst_img)

    replace_all_triangles(
        src_img, dst_img, src_delaunays, dst_delaunays)

    dst_img = apply_seamless_clone(dst_img, dst_img_copy, dst_points)

    return dst_img


def swap_a_face(src_img, dst_img, src_face_points, dst_face_points):
    """
    Swap a single face
    """
    # find the convex hull bounding the landmark points of the images
    src_hulls, dst_hulls = get_convex_hulls(
        src_face_points, dst_face_points)

    swapped_img = swap_using_points(
        src_img, dst_img, src_hulls, dst_hulls)

    return swapped_img


def faceSwap(src_img, dst_img, mode=LARGEST_FACE_MODE, showImages=False):
    """
    Swap a face from src_img to dst_img

    Args:
        src_img: nparray
            Readed value of source image to be swapped on destination image
        dst_img: nparray
            Readed value of destination image
        mode:
            How the faces should be swapped
        showImages : bool
            An optional argument whether or not to show the original images

    Returns:
        nparray:
            The swapped image
    """
    # preprocess images
    src_img, dst_img = preprocess_images(src_img, dst_img)

    # save the original dst img
    dst_img_copy = np.copy(dst_img)

    # find landmark points of the images
    src_face_points = get_face_landmark_points(
        src_img, ARG_IMG_SRC, largest_only=True)

    dst_faces_points = get_face_landmark_points(dst_img, ARG_IMG_DST)

    # if only one face is gonna be replaced it is ultimately largest_face_mode
    dst_faces_num = len(dst_faces_points)
    if dst_faces_num == 1:
        mode = LARGEST_FACE_MODE

    if mode == ALL_FACE_MODE:
        for face_i in range(dst_faces_num):
            dst_face_i_points = dst_faces_points[face_i]
            dst_face_i, crop_box_i = extract_face(dst_img, dst_face_i_points)
            src_face_i, _ = extract_face(src_img, src_face_points)

            swappedImage = swap_a_face(
                src_face_i, dst_face_i, src_face_points, dst_face_i_points)

            dst_img_copy = put_image(
                dst_img_copy, swappedImage, crop_box_i)
    else:
        # find landmark points of the largest face in an image
        large_dst_face_points = choose_largest_face(dst_faces_points)
        dst_face_i, crop_box_i = extract_face(dst_img, large_dst_face_points)

        # match the direction of the two images
        src_img = align_face_direction(
            src_img, src_face_points, large_dst_face_points)

        # recompute lanmark points on the flipped image or new one.
        # ? don't need to recompute, flip points too
        all_src_face_points = get_face_landmark_points(src_img, ARG_IMG_SRC)
        src_face_points = choose_largest_face(all_src_face_points)
        src_face_i, _ = extract_face(src_img, src_face_points)

        swappedImage = swap_a_face(
            src_face_i, dst_face_i, src_face_points, large_dst_face_points)
        dst_img_copy = put_image(
            dst_img_copy, swappedImage, crop_box_i)

    if showImages == True:
        show_images(src_img, dst_img_copy, swappedImage,
                    showOriginalImages=True)

    return swappedImage


def show_images(img1, img2, swappedImage, showOriginalImages=False):
    """
    Display image1, image2 and warped image.

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


if __name__ == '__main__':
    image1 = "faceswap/images/kalise.jpg"
    image2 = "faceswap/images/black_and_white.jpg"

    src_img = cv2.imread(image1)
    dst_img = cv2.imread(image2)

    swappedImage = faceSwap(src_img, dst_img, mode="apply_on_all",
                            showImages=True)
