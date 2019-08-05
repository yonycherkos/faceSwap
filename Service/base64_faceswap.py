import cv2
import magic
import tempfile
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


def np_img_to_base64(np_img):
    # change from numpy array to base64 string(rather than writing to file and read it)
    pil_img = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    np_img_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return np_img_string


def byte_to_img(input_image):
    binary_image = base64.b64decode(input_image)
    file_format = magic.from_buffer(
        base64.b64decode(input_image), mime=True).split('/')[1]

    f = tempfile.NamedTemporaryFile(suffix='*.' + str(file_format))
    f.write(binary_image)
    image = Image.open(f.name).convert('RGB')

    return image


def base64_face_swap(input_image, meme_image, mode):
    inputImage = np_img_from_base64(input_image)
    memeImage = np_img_from_base64(meme_image)

    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
    memeImage = cv2.cvtColor(memeImage, cv2.COLOR_BGR2RGB)

    fs = FaceSwap()
    result = fs.faceSwap(inputImage, memeImage, mode=mode)

    cv2.imwrite("output.jpg", result)

    with open("output.jpg", 'rb') as f:
        img = f.read()
        image = base64.b64encode(img).decode('utf-8')

    return image
