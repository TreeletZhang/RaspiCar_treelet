from concurrent import futures

import grpc
import numpy as np

import raspi_message_pb2
import raspi_message_pb2_grpc
from numproto import ndarray_to_proto, proto_to_ndarray

import matplotlib.pyplot as plt
import keyboard
import os
from PPO import PPO

#####################  hyper parameters  ####################
ENV_ID = 'CarVerification-1'  # environment id
ALG_NAME = 'PPO'


class RaspiMessageServicer(raspi_message_pb2_grpc.RaspiMessageServicer):
    def Init(self, request: raspi_message_pb2.InitRequest, context):
        vis_shape = request.observation_shapes[0].dim  # [48, 64, 3]
        # print("vis_shape:", vis_shape)
        vec_shape = request.observation_shapes[1].dim  # [1]
        # print("vec_shape:", vec_shape)

        try:
            fig, ax = plt.subplots()
            self.im = ax.imshow(np.zeros(request.observation_shapes[0].dim))
        except Exception as e:
            print(e)  # [1]

        self.d_action_dim = request.action_shape.dim
        print('action shape:', self.d_action_dim)  # [5]

        # 加载模型
        try:
            self.agent = PPO(tuple(vis_shape), self.d_action_dim[0], None, 1)  # 传入s.dim的tuple形式
            print("instantiate agent success:", self.agent)
            self.agent.load(ALG_NAME, ENV_ID)
            print("agent loaded")
            print("actor:", self.agent.actor)
        except Exception as e:
            print("exception:", e)

        return raspi_message_pb2.Empty()


    # 使用加载的预训练强化学习模型产生action，返回给树莓派
    def GetAction(self, request: raspi_message_pb2.ObservationsRequest, context):
        vis = proto_to_ndarray(request.observations[0])
        vec = proto_to_ndarray(request.observations[1])

        self.im.set_data(vis)
        plt.pause(0.5)
        distance = vec[0]
        print('distance:', distance)

        state = vis
        d_action_to_raspicar = self.agent.get_action(state, self.d_action_dim[0])
        print("d_action_to_raspicar:", d_action_to_raspicar)
        # print(type(d_action_to_raspicar))  # ndarray
        return raspi_message_pb2.ActionResponse(action=ndarray_to_proto(d_action_to_raspicar))  # 传入的要是numpy



    # 键盘产生动作，返回给树莓派
    # def GetAction(self, request: raspi_message_pb2.ObservationsRequest, context):
    #     vis = proto_to_ndarray(request.observations[0])
    #     # print("shape:", vis.shape)  # （48，64，3）
    #     vec = proto_to_ndarray(request.observations[1])
    #
    #     self.im.set_data(vis)
    #     plt.pause(0.2)
    #
    #     distance = vec[0]
    #     print('distance:', distance)
    #     # ultrasound_direction = np.random.randint(45, 135)
    #     # ultrasound_direction = 90
    #     # camera_direction_h = 90
    #     # camera_direction_v = 0
    #     forward_speed = 0.
    #     backward_speed = 0.
    #     left_speed = 0.
    #     right_speed = 0.
    #
    #     action = np.zeros((1,), np.float32)
    #     if distance <= 20:
    #         action[0] = 0.
    #     else:
    #         if keyboard.is_pressed('w'):
    #             action[0]=1.
    #         elif keyboard.is_pressed('space'):
    #             action[0]=2.
    #         elif keyboard.is_pressed('a'):
    #             action[0]=3.
    #         elif keyboard.is_pressed('d'):
    #             action[0]=4.
    #         elif keyboard.is_pressed('q'):
    #             action[0]=0.
    #
    #     return raspi_message_pb2.ActionResponse(action=ndarray_to_proto(action))


servicer = RaspiMessageServicer()
server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
raspi_message_pb2_grpc.add_RaspiMessageServicer_to_server(servicer, server)
server.add_insecure_port(f'[::]:8888')
server.start()
server.wait_for_termination()
