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


def byte_to_img(input_image):
    binary_image = base64.b64decode(input_image)
    file_format = magic.from_buffer(
        base64.b64decode(input_image), mime=True).split('/')[1]

    f = tempfile.NamedTemporaryFile(suffix='*.' + str(file_format))
    f.write(binary_image)
    image = Image.open(f.name).convert('RGB')

    return image


def face_swap(input_image, meme_image):
    inputImage = np_img_from_base64(input_image)
    memeImage = np_img_from_base64(meme_image)

    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
    memeImage = cv2.cvtColor(memeImage, cv2.COLOR_BGR2RGB)

    fs = FaceSwap()
    result = fs.faceSwap(inputImage, memeImage, mode="apply_on_all")
    cv2.imwrite("output.jpg", result)

    with open("output.jpg", 'rb') as f:
        img = f.read()
        image = base64.b64encode(img).decode('utf-8')

    return image


# with open('image.jpg', 'rb') as f:
#         img = f.read()
#         image = base64.b64encode(img).decode('utf-8')

# with open('meme.jpg', 'rb') as f:
#         img_m = f.read()
#         meme = base64.b64encode(img_m).decode('utf-8')

# face_swap(image,meme)
