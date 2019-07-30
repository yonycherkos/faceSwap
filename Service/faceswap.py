from inspect import getsourcefile
import os.path
import sys

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

from faceSwap import FaceSwap
from PIL import Image
import numpy as np
import base64
import tempfile
import magic
import cv2


def byte_to_img(input_image):
    binary_image = base64.b64decode(input_image)
    file_format = magic.from_buffer(base64.b64decode(input_image), mime=True).split('/')[1]

    f = tempfile.NamedTemporaryFile(suffix='*.' + str(file_format))
    f.write(binary_image)
    image = Image.open(f.name).convert('RGB')

    return image


def face_swap(input_image,meme_image):
    inputImage = np.array(byte_to_img(input_image))
    memeImage = np.array(byte_to_img(meme_image))

    fs = FaceSwap()
    result = fs.faceSwap(inputImage,memeImage)
    cv2.imwrite("output.jpg",result)

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




    





