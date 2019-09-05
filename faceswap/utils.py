import cv2

# type of modes to apply face swapping
ALL_FACE_MODE = 'apply_on_all'
LARGEST_FACE_MODE = 'choose_largest_face'

# type of image arguments for referring image in error messages
ARG_IMG_SRC = 'face'
ARG_IMG_DST = 'meme'


class GRPCException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return "GRPC Exception {}: {}".format(self.code, self.message)


def extract_image(img, face_points, transform=False, padding=(0.1, 0.1)):
    """
    Crop image based on face_points to extract face
    and transform face points corresponding to the cropped image

    Args:
        img: cvimage
        face_points: landmark points of face in img
        transform: whether to transform face_points to the extracted image coordinates
        padding: x and y padding when extracted

    Returns:
        tuple:
            0: cropped cvimage
            1: cropped image box in [x,y,w,h] format
    """
    x, y, w, h = cv2.boundingRect(face_points)
    
    # compute padding size from given padding scales
    padding_x, padding_y = round(padding[0] * w), round(padding[1] * h)
    
    x, y = x - padding_x, y - padding_y
    w, h = w + 2*padding_x, h + 2*padding_y

    # guard x, y from getting negative
    x = max(x, 0)
    y = max(y, 0)
    
    # guard w, h from getting bigger than image dimensions
    img_w, img_h = img.shape[1::-1]
    max_x, max_y = x + w, y + h
    w = img_w if max_x > img_w else w
    h = img_h if max_y > img_h else h

    if transform:
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
