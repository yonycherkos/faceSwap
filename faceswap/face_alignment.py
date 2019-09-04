import cv2
import math
import numpy as np

from geometry import line_parameters, perpendicular_line, dist
from face_detection import get_face_landmark_points, FACIAL_LANDMARKS_68_IDXS
from utils import ARG_IMG_SRC, extract_image


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


def half_faces(img, face_points):
    """
    Get left and right half face corner points

    Args:
        img: cvimage
        face_points: face landmark points

    Returns:
        0: left face rect corners
        1: right face rect corners
    """
    # get friendly img width and height indexes
    img_w, img_h = img.shape[1::-1]
    img_w, img_h = img_w - 1, img_h - 1

    # indexes of interesting face areas
    (left_start, left_end) = FACIAL_LANDMARKS_68_IDXS['left_eye']
    (right_start, right_end) = FACIAL_LANDMARKS_68_IDXS['right_eye']
    (nose_start, nose_end) = FACIAL_LANDMARKS_68_IDXS['nose_bottom']

    # get representative points of eyes and its line
    left_eye_points = face_points[left_start: left_end]
    right_eye_points = face_points[right_start: right_end]
    left_eye_center = left_eye_points.mean(axis=0).astype('int')
    right_eye_center = right_eye_points.mean(axis=0).astype('int')
    eyes_line = line_parameters(left_eye_center, right_eye_center)

    # get representative points of nose and its line
    bottom_nose_points = face_points[nose_start: nose_end]
    bottom_nose_center = bottom_nose_points.mean(axis=0).astype('int')
    nose_line = perpendicular_line(
        bottom_nose_center, eyes_line[0])

    # split image into half face based on nose line
    nose_m, nose_b = nose_line
    nose_y0_x = round(-nose_b/nose_m).astype('int')
    nose_h_x = round(
        (img.shape[0] - nose_b)/nose_m).astype('int')

    # left half face corners
    left_half_face = np.array([
        (0, 0),
        (nose_y0_x, 0),
        (nose_h_x, img_h),
        (0, img_h)
    ], dtype="float32")

    # right half face corners
    right_half_face = np.array([
        (nose_y0_x, 0),
        (img_w, 0),
        (img_w, img_h),
        (nose_h_x, img_h)
    ], dtype="float32")

    return left_half_face, right_half_face


def align_half_faces(src_img, dst_img, src_face_points, dst_face_points, DEBUG=True):
    """
    Align left and right face using perspective transform
    A total failure for aligning faces

    Args:
        src_img: cvimage
        dst_img: cvimage
        src_face_points: face landmark points of src_img
        dst_face_points: face landmark points of dst_img

    Returns:
        processed src_img which tries to align half faces to dst_img(! failed)
    """
    # get half face corners
    src_left_face, src_right_face = half_faces(src_img, src_face_points)
    dst_left_face, dst_right_face = half_faces(dst_img, dst_face_points)
    orig_dst_left_face, orig_dst_right_face = dst_left_face.copy(), dst_right_face.copy()

    if DEBUG:
        cv2.drawContours(
            src_img, [src_left_face.astype('int')], -1, (255, 0, 0))
        cv2.drawContours(
            src_img, [src_right_face.astype('int')], -1, (255, 0, 0))
        cv2.drawContours(
            dst_img, [dst_left_face.astype('int')], -1, (255, 0, 0))
        cv2.drawContours(
            dst_img, [dst_right_face.astype('int')], -1, (255, 0, 0))

    # extract each half faces
    src_left_img, src_left_box = extract_image(
        src_img, src_left_face, transform=True)
    src_right_img, src_right_box = extract_image(
        src_img, src_right_face, transform=True)

    dst_left_img, dst_left_box = extract_image(
        dst_img, dst_left_face, transform=True)
    dst_right_img, dst_right_box = extract_image(
        dst_img, dst_right_face, transform=True)

    # get transform matrix from src to dst half faces
    left_transfom_matrix = cv2.getPerspectiveTransform(
        src_left_face, dst_left_face)
    right_transform_matrix = cv2.getPerspectiveTransform(
        src_right_face, dst_right_face)

    # get destination sizes for warped half images
    dst_left_max_size = tuple(dst_left_box[2:4])
    dst_right_max_size = tuple(dst_right_box[2:4])

    # warp half faces of src_img to dst_img
    warped_left_face = cv2.warpPerspective(
        src_left_img, left_transfom_matrix, dst_left_max_size)
    warped_right_face = cv2.warpPerspective(
        src_right_img, right_transform_matrix, dst_right_max_size)

    # merge warped half faces to get full face

    img_w, img_h = dst_img.shape[1::-1]

    # set up left face to be merged
    left_img = np.zeros((img_h, img_w, 3), dtype='uint8')
    left_img[:, :warped_left_face.shape[1], :] += warped_left_face
    cv2.fillConvexPoly(
        left_img, orig_dst_right_face.astype('int'), (0, 0, 0))

    # set up right face to be merged
    right_img = np.zeros((img_h, img_w, 3), dtype='uint8')
    right_img[:, img_w-warped_right_face.shape[1]:, :] += warped_right_face
    cv2.fillConvexPoly(
        right_img, orig_dst_left_face.astype('int'), (0, 0, 0))

    # merge half face images
    merged_img = left_img + right_img

    if DEBUG:
        cv2.imshow('left_img', left_img)
        cv2.imshow('right_img', right_img)
        cv2.imshow('merged_img', merged_img)
        cv2.imshow('left_warped', warped_left_face)
        cv2.imshow('right_warped', warped_right_face)
        cv2.imshow('stest', src_img)
        cv2.imshow('dtest', dst_img)
        cv2.waitKey()

    return merged_img


def get_half_face_proportions(face_points):
    """
    Get left and right half face size proportions

    Args:
        face_points: face landmark points

    Returns:
        0: from left eye to mid of the eyes size proportion
        1: from right eye to mid of the eyes size proportion
    """

    # indexes of interesting face areas
    (left_start, left_end) = FACIAL_LANDMARKS_68_IDXS['left_eye']
    (right_start, right_end) = FACIAL_LANDMARKS_68_IDXS['right_eye']
    mid_eye_index = FACIAL_LANDMARKS_68_IDXS['nose_top']

    # get representative points of eyes and its line
    left_eye_points = face_points[left_start: left_end]
    right_eye_points = face_points[right_start: right_end]
    left_eye_center = left_eye_points.mean(axis=0).astype('int')
    right_eye_center = right_eye_points.mean(axis=0).astype('int')

    # get representative middle point between eyes
    mid_eyes_point = face_points[mid_eye_index]

    # compute distance between eyes
    eyes_dist = dist(left_eye_center, right_eye_center)

    # compute distance percentage between middle of eyes to each eye
    left_eye_dist = dist(left_eye_center, mid_eyes_point) / eyes_dist
    right_eye_dist = dist(right_eye_center, mid_eyes_point) / eyes_dist

    return left_eye_dist, right_eye_dist


def align_src_img_direction(src_img, src_face_points, dst_face_points):
    """
    Flip src_img vertically based on face direction(left or right) to align faces or not

    Args:
        src_img: cvimage
        src_face_points: face landmark points of src_img
        dst_face_points: face landmark points of dst_img

    Returns:
        0: flipped src_img if necessary or the original
        1: flipped src_face_points if necessary or the original
    """
    # get left and right face proportions
    src_left_size, src_right_size = get_half_face_proportions(
        src_face_points)
    dst_left_size, dst_right_size = get_half_face_proportions(
        dst_face_points)

    # which face side is smaller
    small_src_left = src_left_size < 0.50
    small_dst_left = dst_left_size < 0.50

    # flip image if face sides are different sizes
    if small_src_left != small_dst_left:
        src_img = cv2.flip(src_img, flipCode=1)
        src_face_points = get_face_landmark_points(
            src_img, ARG_IMG_SRC, largest_only=True)[0]
        # ! avoids negative landmark points
        # ! exist only because of finding face on extracted face image
        # ! approximate negative to zero
        src_face_points[src_face_points < 0] = 0

    return src_img, src_face_points


def flip_points(img, points):
    """
    Flip points on img based on vertical half line of img
    """
    # get half width value of img
    w = img.shape[1]
    half_w = round(w/2)

    # set up half width points
    half_points = np.full(
        (points.shape[0]), half_w, dtype=points.dtype)

    # compute the flip in x points using half width points
    points[:, 0] = half_points - \
        (points[:, 0] - half_points)

    return points


def align_horizontal(src_img, src_face_points, dst_face_points):
    """
    Rotates image to align src_face eyes to dst_face eyes
    Changes src_face_points to rotated points N.B but could get to negative!!

    Args:
        src_img: cvimage
        src_face_points: src face landmark points
        dst_face_points: dst face landmark points

    Returns:
        0: rotated cvimage
        1: rotated src_face_points(could be negative!!) ! fix this
    """
    # indexes of interesting face areas
    (left_start, left_end) = FACIAL_LANDMARKS_68_IDXS['left_eye']
    (right_start, right_end) = FACIAL_LANDMARKS_68_IDXS['right_eye']

    # get representative points of eyes for src_img
    src_left_eye_points = src_face_points[left_start:left_end]
    src_right_eye_points = src_face_points[right_start:right_end]
    src_left_eye_center = src_left_eye_points.mean(axis=0).astype('int')
    src_right_eye_center = src_right_eye_points.mean(axis=0).astype('int')

    # get representative points of eyes for dst_img
    dst_left_eye_points = dst_face_points[left_start:left_end]
    dst_right_eye_points = dst_face_points[right_start:right_end]
    dst_left_eye_center = dst_left_eye_points.mean(axis=0).astype('int')
    dst_right_eye_center = dst_right_eye_points.mean(axis=0).astype('int')

    # compute eye line vectors
    src_eye_vector = (
        src_left_eye_center[0] - src_right_eye_center[0],
        src_left_eye_center[1] - src_right_eye_center[1]
    )
    dst_eye_vector = (
        dst_left_eye_center[0] - dst_right_eye_center[0],
        dst_left_eye_center[1] - dst_right_eye_center[1]
    )

    # compute angle using dot product rule
    inner_product = src_eye_vector[0] * dst_eye_vector[0] + \
        src_eye_vector[1] * dst_eye_vector[1]
    src_norm = math.hypot(src_eye_vector[0], src_eye_vector[1])
    dst_norm = math.hypot(dst_eye_vector[0], dst_eye_vector[1])

    angle = math.degrees(math.acos(inner_product/(src_norm*dst_norm)))

    # compute rotation center which is middle point between eyes
    # ! not right point actually(need to fix this)
    rotation_center = (
        (src_left_eye_center[0] + src_right_eye_center[0]) // 2,
        (src_right_eye_center[1] + src_right_eye_center[1]) // 2
    )

    print('Angle: ', angle)
    print('Rotation center: ', rotation_center)

    # rotate src_img based on angle
    M = cv2.getRotationMatrix2D(rotation_center, -angle, 1.0)
    rotated_src_img = cv2.warpAffine(
        src_img, M, src_img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # transform landmark points as same as image rotation
    src_face_points = np.dot(
        M,
        np.vstack(
            (src_face_points.T, np.ones((1, 68)))
        )
    ).T.astype('int')

    return rotated_src_img, src_face_points
