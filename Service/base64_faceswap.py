import cv2
# import magic
# import tempfile
import base64
import numpy as np
from PIL import Image
from faceSwap import FaceSwap
from inspect import getsourcefile
import os.path
import sys
import io

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)


def np_img_from_base64(image_base64):
    img_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(img_bytes))
    return np.array(image)


def base64_from_np_img(np_img):
    # change from numpy array to base64 string(rather than writing to file and read it)
    retval, buffer = cv2.imencode('.jpg', np_img)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text


def base64_face_swap(input_image, meme_image, mode):
    inputImage = np_img_from_base64(input_image)
    memeImage = np_img_from_base64(meme_image)

    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
    memeImage = cv2.cvtColor(memeImage, cv2.COLOR_BGR2RGB)

    fs = FaceSwap()
    result = fs.faceSwap(inputImage, memeImage, mode=mode)

    image = base64_from_np_img(result)

    return image