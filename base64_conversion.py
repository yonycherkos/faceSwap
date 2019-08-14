import cv2
import base64
import numpy as np
from PIL import Image
import io


def np_img_from_base64(image_base64):
    img_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(img_bytes))

    bgr_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return bgr_image


def base64_from_np_img(np_img):
    # change from numpy array to base64 string(rather than writing to file and read it)
    retval, buffer = cv2.imencode('.jpg', np_img)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text
