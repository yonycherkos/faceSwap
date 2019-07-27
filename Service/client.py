import grpc
from concurrent import futures
import time

import os
import grpc

import image_swap_pb2
import image_swap_pb2_grpc

from inspect import getsourcefile
import os.path
import sys
import base64
import magic
from PIL import Image
import tempfile
import cv2

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

import argparse

parser = argparse.ArgumentParser()






class ClientTest():
    def __init__(self, port='localhost:50051'):
        self.port = port

    def open_grpc_channel(self):
        channel = grpc.insecure_channel(self.port)
        stub = image_swap_pb2_grpc.FaceSwapStub(channel)
        return stub

    def send_request(self, stub, image_input,meme_input):
        images = image_swap_pb2.ImageFileIn(input_image=image_input,meme_image=meme_input)

        response = stub.faceSwap(images)

        return response

    def byte_to_img(self,input_image):
        binary_image = base64.b64decode(input_image)
        file_format = magic.from_buffer(base64.b64decode(input_image), mime=True).split('/')[1]

        f = tempfile.NamedTemporaryFile(suffix='.jpg')
        f.write(binary_image)
        image = Image.open(f.name).convert('RGB')

        return image

    def close_channel(self, channel):
        pass


if __name__ == "__main__":
    with open('image.jpg', 'rb') as f:
        img = f.read()
        image = base64.b64encode(img).decode('utf-8')
    with open('meme.jpg', 'rb') as f:
        img_m = f.read()
        meme = base64.b64encode(img_m).decode('utf-8')

    client_test = ClientTest()
    stub = client_test.open_grpc_channel()
    response = client_test.send_request(stub,image_input=image,meme_input=meme)


    image = client_test.byte_to_img(response.image_out)
    image.save('a.jpg')
    

