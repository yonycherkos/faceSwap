# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import image_swap_pb2 as image__swap__pb2


class FaceSwapStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.faceSwap = channel.unary_unary(
        '/FaceSwap/faceSwap',
        request_serializer=image__swap__pb2.ImageFileIn.SerializeToString,
        response_deserializer=image__swap__pb2.ImageFileOut.FromString,
        )


class FaceSwapServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def faceSwap(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_FaceSwapServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'faceSwap': grpc.unary_unary_rpc_method_handler(
          servicer.faceSwap,
          request_deserializer=image__swap__pb2.ImageFileIn.FromString,
          response_serializer=image__swap__pb2.ImageFileOut.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'FaceSwap', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))