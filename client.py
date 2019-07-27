
import requests
import json
import base64
import tempfile
import cv2

base_image_path = "images/original_images/yony.jpg"
meme_image_path = "images/original_images/anchorman.jpg"


# def image_to_str(image_path):
#     with open(image_path, 'rb') as f:
#         img = f.read()
#         image_64 = base64.b64encode(img).decode('utf-8')
#
#     binary_image = base64.b64decode(image_64)
#     return binary_image

def img_to_str_encoder(image_path):
    with open(image_path, 'rb') as f:
        img = f.read()
        img_base64 = base64.b64encode(img)

    return img_base64


def str_to_img_decoder(img_base64):

    img = base64.b64decode(img_base64)

    return img


# def save_image(image, name, image_ext):
#     file_name = name + image_ext
#
#     f = tempfile.NamedTemporaryFile()
#     f.write(binary_image)
#
#     img = cv2.imread(f.name)
#     return result, img


def send_request():
    url = 'http://127.0.0.1:5000/image-swap/'

    base_img_b64 = img_to_str_encoder(base_image_path)
    meme_img_b64 = img_to_str_encoder(meme_image_path)

    files = {'base_img_b64': base_img_b64,
             'meme_img_b64': meme_img_b64}
    response = requests.post(url, files=json.dumps(files))
    print(type(response))


def recieve_request():
    url = 'http://127.0.0.1:5000/image-swap/'

    response = requests.get(url)

    return str_to_img_decoder(response.content)


if __name__ == "__main__":

    send_request()
    # swappedImage = recieve_request()
