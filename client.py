import grpc
import numpy as np

import raspi_message_pb2
import raspi_message_pb2_grpc
from numproto import ndarray_to_proto, proto_to_ndarray


channel = grpc.insecure_channel('58.199.162.110:8888')
stub = raspi_message_pb2_grpc.RaspiMessageStub(channel)
response = stub.Init(raspi_message_pb2.InitRequest(
    observation_shapes=[raspi_message_pb2.Shape(dim=[1, 2]),
                        raspi_message_pb2.Shape(dim=[1, 2, 3])],
    action_shape=raspi_message_pb2.Shape(dim=[1, 10])
))

while True:
    input()
    response = stub.GetAction(raspi_message_pb2.ObservationsRequest(
        observations=[
            ndarray_to_proto(np.random.randn(10, 5)),
            ndarray_to_proto(np.random.randn(10, 2))
        ]
    ))

    print(proto_to_ndarray(response.action))
