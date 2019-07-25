
# step-1: imread
# step-2: img -> str(encode)
# step-3: send
# step-4: recv

import requests
import json
import base64


def image_encoding(image):
    with open(image, 'rb') as file:
        img = file.read()
        encoded_img = base64.b64encode(img).decode('utf-8')

    binary_img = base64.b64decode(encoded_img)

    return binary_img


def send_image(url, image1, image2):
    # url = "http://localhost:5000/"

    # read and encode an images
    binary_img1 = image_encoding(image1)
    binary_img2 = image_encoding(image2)

    # send the encoded images
    payload = {"encoded_img1": binary_img1, "encoded_img2": binary_img2}
    requests.post(url, payload)


def recieve_image(url):

    response = requests.get(url)

    return response.content


if __name__ == "__main__":

    url = "http://127.0.0.1:5000/"
    image1 = "/home/yonathan/Documents/projects/faceSwap/images/original_images/yony.jpg"
    image2 = "/home/yonathan/Documents/projects/faceSwap/images/original_images/anchorman.jpg"
    send_image(url, image1, image2)
