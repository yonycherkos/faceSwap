

# step-1: recv - str
# step-2: str -> np.array()(decode)
# stpe-3: swap
# stpe-4: send

import flask
from faceSwap import FaceSwap
from flask import Flask, request, make_response, jsonify
import base64
import tempfile
import cv2


app = flask.Flask(__name__)
app.config["DEBUG"] = True


# def save_image(image, name, image_ext):
#     file_name = name + image_ext
#
#     f = tempfile.NamedTemporaryFile()
#     f.write(binary_image)
#
#     img = cv2.imread(f.name)
#     return result, img

def img_to_str_encoder(image_path):
    with open(image_path, 'rb') as f:
        img = f.read()
        img_base64 = base64.b64encode(img)

    return img_base64


def str_to_img_decoder(img_base64):

    img = base64.b64decode(img_base64)

    return img


def save_image(file, file_name, file_ext):

    with open(file_name + "." + file_ext, "wb") as p:
        p.write(str_to_img_decoder(file))

    img = cv2.imread(file_name + "." + file_ext)

    return img


# def image_to_str(image_path):
#     with open(image_path, 'rb') as f:
#         img = f.read()
#         image_64 = base64.b64encode(img).decode('utf-8')
#
#     binary_image = base64.b64decode(image_64)
#     return binary_image


@app.route('/image-swap/', methods=['POST'])
def face_swap():

    base_img_b64 = request.files['base_img_b64']
    base_img_filename = base_img_b64.filename
    base_img = save_image(base_img_b64, base_img_filename, "jpg")

    meme_img_b64 = request.files['meme_img_b64']
    meme_img_fileame = meme_img_b64.filename
    meme_img = save_image(meme_img_b64, meme_img_filename, "jpg")

    fs = FaceSwap()
    result = fs.faceSwap(base_img, meme_img)

    # res = make_response()
    #
    # res = result
    # return res
    print(result.any())

    if result.any():
        try:
            return send_file("output.jpg", attachment_filename='output.jpg')
            # return image_to_str(result)
        except Exception as e:
            print("exception raises")
            print(str(e))
            return str(e)
    else:
        return "Unable to Genrate meme"


if __name__ == "__main__":
    app.run()
