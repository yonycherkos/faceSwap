import cv2
import dlib
import numpy as np
import grpc
from collections import OrderedDict

from utils import GRPCException

# model file name for face landmark detection in dlib library
FACE_LANDMARK_SHAPE_DETECTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'

NUMBER_OF_FACE_LANDMARKS = 68   # number of facelandmark points

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("nose_bottom", (31, 36)),
    ("jaw", (0, 17)),
    ("nose_top", (27))
])


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
            list of landmark points for every face in a given image as request
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
        # to accomodate interface make return_value list
        faces_landmark_points = [faces_landmark_points]
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
