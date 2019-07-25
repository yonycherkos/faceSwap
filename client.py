import requests, json
import base64
import tempfile
import cv2



def image_to_str(image_path):
    with open(image_path, 'rb') as f:
        img = f.read()
        image_64 = base64.b64encode(img).decode('utf-8')


    binary_image = base64.b64decode(image_64)
    return binary_image

def save_image(image,name,image_ext):
    file_name = name+image_ext

    f = tempfile.NamedTemporaryFile()
    f.write(binary_image)

    img = cv2.imread(f.name)
    return result,img



def send_request():
    url = 'http://127.0.0.1:5000/image-swap/'
    files = {'base_image': open('image.jpg', 'rb'),'meme_image':open('meme.jpg', 'rb')}
    responce = requests.post(url, files=files)
    print(type(responce))

    



send_request()
