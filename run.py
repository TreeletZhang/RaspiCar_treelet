import time

import grpc
import numpy as np

import raspi_message_pb2
import raspi_message_pb2_grpc
from numproto import ndarray_to_proto, proto_to_ndarray

from raspi_car import RaspiCar


VISUAL_SCALE = 0.1
DIRECTION_MIN_INTERVAL = 0.2

if __name__ == "__main__":
    car = RaspiCar()

    vis = car.camera_observe(VISUAL_SCALE)

    if vis is None:
        print('No camera')
        exit()

    channel = grpc.insecure_channel('58.199.160.125:8888',
                                    [('grpc.max_reconnect_backoff_ms', 5000)])
    stub = raspi_message_pb2_grpc.RaspiMessageStub(channel)

    while True:
        try:
            # observations: [image, vector]
            # action: vector
            # 小车请求server的init函数，得到responce
            response = stub.Init(raspi_message_pb2.InitRequest(
                observation_shapes=[raspi_message_pb2.Shape(dim=list(vis.shape)),
                                    raspi_message_pb2.Shape(dim=[1, ])],
                action_shape=raspi_message_pb2.Shape(dim=[5, ])
            ))
        except Exception as e:
            print('Init disconnected')
            # print(e)
            continue

        start = time.time()

        while True:
            vis = car.camera_observe(VISUAL_SCALE).astype(np.float32)
            distance = car.ultrasound_distance()
            vec = np.array([distance], np.float32)
            # 小车请求server的GetAction函数，发送当前obs,得到 action responce
            try:
                response = stub.GetAction(raspi_message_pb2.ObservationsRequest(
                    observations=[
                        ndarray_to_proto(vis),
                        ndarray_to_proto(vec)
                    ]
                ))
            except Exception as e:
                print('GetAction disconnected')
                break

            action = proto_to_ndarray(response.action)
            print("action:", action)  #[0,0,0,0]


            # speed, ultrasound_direction, camera_direction_h, camera_direction_v = action
            # forward_speed, backward_speed, left_speed, right_speed = action
            # print('f_s:',forward_speed)
            # print('b_s:',backward_speed)
            # print('l_s:',left_speed)
            # print('r_s:',right_speed)
            # car.forward(forward_speed)

            # if time.time() - start >= DIRECTION_MIN_INTERVAL:
            #     car.ultrasound_direction(ultrasound_direction)
            #     car.camera_direction(camera_direction_h, camera_direction_v)
            #     start = time.time()

            # if time.time() - start >= DIRECTION_MIN_INTERVAL:
            #     if backward_speed !=0:
            #         car.backward(backward_speed)
            #     if left_speed != 0:
            #         car.left(left_speed)
            #     if right_speed != 0:
            #         car.right(right_speed)
            #     start = time.time()

            if time.time() - start >= DIRECTION_MIN_INTERVAL:
                if action[0]==1:
                    car.forward(15)
                elif action[0]==2:
                    car.backward(15)
                elif action[0]==3:
                    car.left(20)
                elif action[0]==4:
                    car.right(20)
                elif action[0]==0:
                    car.stop()
                start = time.time()
