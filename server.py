
# step-1: recv - str
# step-2: str -> np.array()(decode)
# stpe-3: swap
# stpe-4: send

from flask import Flask, request, Response
import requests
import numpy as np
import faceSwap

app = Flask(__name__)


def recieve_image(url):

    response = requests.get(url).json()
    encoded_img1 = response['encoded_img1']
    encoded_img2 = response['encoded_img2']

    img1 = base64.b64decode(encoded_img1)
    img2 = base64.b64decode(encoded_img2)

    img1 = np.array(img1)
    img2 = np.array(img2)

    return img1, img2

# define the basic route / and corresponding request hadler
@app.route("/")
def send_image():

    img1, img2 = recieve_image(url)

    faceSwap = faceSwap.FaceSwap()
    swapped_image = faceSwap.faceSwap(img1, img2)

    encoded_swapped_image = base64.b64encode(swappedImage)

    return "hello"


if __name__ == "__main__":
    app.run()
