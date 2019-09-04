import grpc
from concurrent import futures
import time

import image_swap_pb2
import image_swap_pb2_grpc

from base64_conversion import np_img_from_base64, base64_from_np_img
from faceswap.utils import GRPCException
from faceswap.face_swapping import swap_faces


class FaceSwapServicer(image_swap_pb2_grpc.FaceSwapServicer):
    def faceSwap(self, request, context):
        if request.input_image is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Input Image is required.")
            return image_swap_pb2.ImageFileOut()
        if request.meme_image is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Meme Image is required.")
            return image_swap_pb2.ImageFileOut()

        mode = 'choose_largest_face' if request.mode is None else request.mode

        # change base64 images to cv2 computable images
        inputImage = np_img_from_base64(request.input_image)
        memeImage = np_img_from_base64(request.meme_image)

        # swap faces
        try:
            result = swap_faces(inputImage, memeImage, mode=mode)
        except GRPCException as rpc_exception:
            # custom Exception to respond for application specific errors
            context.set_code(rpc_exception.code)
            context.set_details(rpc_exception.message)
            return image_swap_pb2.ImageFileOut()
        except Exception as python_exception:
            # catch all to avoid returning exception 3 times for client(wierd)
            print('Exception: ', repr(python_exception))

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Server Error. Please, try again")
            return image_swap_pb2.ImageFileOut()

        # change result image to base64 for response
        result_base64 = base64_from_np_img(result)

        response = image_swap_pb2.ImageFileOut(image_out=result_base64)

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
