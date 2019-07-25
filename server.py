import flask
from faceSwap import FaceSwap
from flask import Flask, request, make_response, jsonify
import base64
import tempfile
import cv2


app = flask.Flask(__name__)
app.config["DEBUG"] = True

def save_image(image,name,image_ext):
    file_name = name+image_ext

    f = tempfile.NamedTemporaryFile()
    f.write(binary_image)

    img = cv2.imread(f.name)
    return result,img

def image_to_str(image_path):
    with open(image_path, 'rb') as f:
        img = f.read()
        image_64 = base64.b64encode(img).decode('utf-8')


    binary_image = base64.b64decode(image_64)
    return binary_image






@app.route('/image-swap/', methods=['POST'])
def face_swap():
    file = request.files['base_image']
    filename = file.filename
    file.save("base_img.jpg")

    file2 = request.files['meme_image']
    filename2 = file2.filename
    file2.save("meme_img.jpg")


    fs = FaceSwap("base_img.jpg","meme_img.jpg")
    result = fs.faceSwap()
    if result:
        try:
            return send_file("output.jpg", attachment_filename='output.jpg')
        except Exception as e:
            return str(e)
    else:
        return "Unable to Genrate meme"
   

        
        



    

    




app.run()