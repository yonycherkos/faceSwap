import numpy as np
import cv2
from PIL import Image

from face_detection import get_face_landmark_points
from face_alignment import align_src_img_direction
from utils import LARGEST_FACE_MODE, ARG_IMG_SRC, ARG_IMG_DST, \
    extract_face, put_image


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


def replace_triangle(src_img, dst_img, src_tri_points, dst_tri_points, DEBUG=True):
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
    assert(all(src_img_roi.shape))
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
    mask = np.zeros(src_img.shape, dtype=src_img.dtype)
    cv2.fillConvexPoly(mask, np.int32(src_points), (255, 255, 255))

    # ? check MIXED_CLONE mode
    swapped_image = cv2.seamlessClone(
        src_img, dst_img, mask, dst_points_center, cv2.NORMAL_CLONE)
    if DEBUG:
        cv2.imshow('swap', swapped_image)
        cv2.waitKey(0)

    return swapped_image


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


def swap_a_face(src_img, dst_img, src_face_points, dst_face_points, use_scale=True):
    """
    Swap a single face
    Considers image contains extracted face
    """

    # scale the face images to match before any preprocessing
    if use_scale:
        dst_img_h = dst_img.shape[0]
        dst_img_w = dst_img.shape[1]
        src_img = cv2.resize(src_img, (dst_img_w, dst_img_h))
        src_face_points = get_face_landmark_points(
            src_img, ARG_IMG_SRC, largest_only=True)[0]

    # align src_img face direction if necessary
    align_src_img_direction(src_img, src_face_points, dst_face_points)

    # find the convex hull bounding the landmark points of the images
    src_hulls, dst_hulls = get_convex_hulls(
        src_face_points, dst_face_points)

    swapped_img = swap_using_points(
        src_img, dst_img, src_hulls, dst_hulls)

    return swapped_img


def swap_faces(src_img, dst_img, mode=LARGEST_FACE_MODE, showImages=False):
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

    # find landmark points of src
    src_face_points = get_face_landmark_points(
        src_img, ARG_IMG_SRC, largest_only=True)[0]

    # find landmark points of dst
    largest_dst_only = True if mode == LARGEST_FACE_MODE else False
    dst_faces_points = get_face_landmark_points(
        dst_img, ARG_IMG_DST, largest_only=largest_dst_only)

    src_face_img, _ = extract_face(src_img, src_face_points)

    for dst_face_points in dst_faces_points:
        dst_face_img, face_box = extract_face(dst_img, dst_face_points)

        swapped_img = swap_a_face(
            src_face_img, dst_face_img, src_face_points, dst_face_points)

        dst_img_copy = put_image(
            dst_img_copy, swapped_img, face_box)

    if showImages == True:
        show_images(src_img, dst_img_copy, swapped_img,
                    showOriginalImages=True)

    return swapped_img


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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--src_img', default='kalise.jpg')
    ap.add_argument('-d', '--dst_img', default='black_and_white.jpg')
    args = ap.parse_args()

    base_folder = 'faceswap/images/'
    image1 = base_folder + args.src_img
    image2 = base_folder + args.dst_img

    src_img = cv2.imread(image1)
    dst_img = cv2.imread(image2)

    swappedImage = swap_faces(src_img, dst_img, mode="apply_on_all",
                              showImages=True)
