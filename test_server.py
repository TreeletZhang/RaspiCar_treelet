from PPO import PPO
import numpy as np

ENV_ID = 'CarVerification-1'
ALG_NAME = 'PPO'

class Server():
    def __init__(self):
        self.obs_shape = (48, 64, 3)
        self.d_action_dim = [5]
        self.agent = PPO(self.obs_shape, self.d_action_dim[0], None, 1)
        print("instantiate agent success:", self.agent)
        self.agent.load(ALG_NAME, ENV_ID)
        print("agent loaded")
        print("actor:", self.agent.actor)
        print("actor_weights:", self.agent.actor.all_weights)

    def GetAction(self, state):
        action = self.agent.get_action(state, self.d_action_dim[0])
        d_action_to_raspicar = action[0]
        return d_action_to_raspicar


if __name__ == '__main__':
    server = Server()
    for i in range(10):
        state = np.random.rand(48, 64, 3)
        a = server.GetAction(state)
        print("a:", a)
