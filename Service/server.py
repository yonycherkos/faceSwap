import grpc
from concurrent import futures
import time

import image_swap_pb2
import image_swap_pb2_grpc

from base64_faceswap import base64_face_swap


class FaceSwapServicer(image_swap_pb2_grpc.FaceSwapServicer):
    def faceSwap(self, request, context):
        if request.input_image is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Input Image is required")
            return image_swap_pb2.ImageFileOut()
        if request.meme_image is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Meme Image is required")
            return image_swap_pb2.ImageFileOut()

        mode = 'choose_largest_face' if request.mode is None else request.mode

        result = base64_face_swap(
            request.input_image, request.meme_image, mode)
        response = image_swap_pb2.ImageFileOut(image_out=result)

        return response


class Server():
    def __init__(self):
        self.port = '[::]:50051'
        self.server = None

    def start_server(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        image_swap_pb2_grpc.add_FaceSwapServicer_to_server(
            FaceSwapServicer(), self.server)
        print('Starting server. Listening on port 50051.')
        self.server.add_insecure_port(self.port)
        self.server.start()

    def stop_server(self):
        self.server.stop(0)


if __name__ == '__main__':
    server = Server()
    server.start_server()

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop_server()
