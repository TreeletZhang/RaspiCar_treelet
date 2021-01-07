"""
Proximal Policy Optimization (PPO)
So far there's only PPO-Clip
----------------------------
A simple version of Proximal Policy Optimization (PPO) using single thread.
PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.
PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.
Reference
---------
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials
Environment
-----------
Unity env, continual or discrete action space
Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0
To run
------
python tutorial_PPO.py --train/test
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl

#####################  hyper parameters  ####################

RANDOM_SEED = 1  # random seed
TRAIN_EPISODES = 3000  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for testing
MAX_STEPS = 200  # total number of steps for each episode
GAMMA = 0.9  # reward discount
LR_A = 0.000008  # learning rate for actor0.00001
LR_C = 0.00008  # learning rate for critic0.0005
BATCH_SIZE = 32  # update batch size
ACTOR_UPDATE_STEPS = 10  # actor update steps
CRITIC_UPDATE_STEPS = 10  # critic update steps

# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

# ppo-clip parameters
EPSILON = 0.2


###############################  PPO  ####################################


class PPO(object):
    """
    PPO class
    """
    def __init__(self, state_dim, d_action_dim, c_action_dim, action_bound, method='clip'):
        # critic
        with tf.name_scope('critic'):
            if len(state_dim) >= 3:  # 图像输入
                visual_inputs = tl.layers.Input([None, *state_dim], tf.float32, 'state')
                visual_embedding1 = tl.layers.Conv2dLayer(shape=(3, 3, 3, 32), strides=(1, 1, 1, 1), act=tf.nn.relu, name='conv2d_1')(visual_inputs)
                visual_embedding2 = tl.layers.Conv2dLayer(shape=(5, 5, 32, 64), strides=(1, 1, 1, 1), act=tf.nn.relu, name='conv2d_2')(visual_embedding1)
                # shape设置卷积核的形状，前两维是filter的大小，第三维表示前一层的通道数，也即每个filter的通道数，第四维表示filter的数量
                inputs = tl.layers.Flatten()(visual_embedding2)
                layer = tl.layers.Dense(64, tf.nn.relu)(inputs)
                layer = tl.layers.Dense(64, tf.nn.relu)(layer)
                v = tl.layers.Dense(1)(layer)
                self.critic = tl.models.Model(visual_inputs, v)
                self.critic.train()
            else:  # 向量输入
                inputs = tl.layers.Input([None, *state_dim], tf.float32, 'state')
                layer = tl.layers.Dense(64, tf.nn.relu)(inputs)
                layer = tl.layers.Dense(64, tf.nn.relu)(layer)
                v = tl.layers.Dense(1)(layer)
                self.critic = tl.models.Model(inputs, v)
                self.critic.train()

        # actor
        with tf.name_scope('actor'):
            if len(state_dim) >= 3:  # 图像输入
                visual_inputs = tl.layers.Input([None, *state_dim], tf.float32, 'state')
                visual_embedding1 = tl.layers.Conv2dLayer(shape=(3, 3, 3, 32), strides=(1, 1, 1, 1), act=tf.nn.relu, name='conv2d_1')(visual_inputs)
                visual_embedding2 = tl.layers.Conv2dLayer(shape=(5, 5, 32, 64), strides=(1, 1, 1, 1), act=tf.nn.relu, name='conv2d_2')(visual_embedding1)
                # shape设置卷积核的形状，前两维是filter的大小，第三维表示前一层的通道数，也即每个filter的通道数，第四维表示filter的数量
                inputs = tl.layers.Flatten()(visual_embedding2)
            else:  # 向量输入
                inputs = tl.layers.Input([None, *state_dim], tf.float32, 'state')
            layer = tl.layers.Dense(64, tf.nn.relu)(inputs)
            layer = tl.layers.Dense(64, tf.nn.relu)(layer)

            if d_action_dim != 0:  # 离散动作
                all_act = tl.layers.Dense(d_action_dim, tf.nn.relu)(layer)
                #all_act = tf.nn.softmax(act_embedding)
                self.actor = tl.models.Model(visual_inputs, all_act)  # Model()的传入要是layers才行
                self.actor.train()
            else:  # 连续动作
                a = tl.layers.Dense(c_action_dim, tf.nn.tanh)(layer)
                mean = tl.layers.Lambda(lambda x: x * action_bound, name='lambda')(a)
                logstd = tf.Variable(np.zeros(c_action_dim, dtype=np.float32))
                self.actor = tl.models.Model(inputs, mean)
                self.actor.trainable_weights.append(logstd)
                self.actor.logstd = logstd
                self.actor.train()

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

        self.method = method
        if method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif method == 'clip':
            self.epsilon = EPSILON

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        self.action_bound = action_bound

    def train_actor_continuous(self, state, action, adv, old_pi):
        """
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return: kl_mean or None
        """
        with tf.GradientTape() as tape:
            mean, std = self.actor(state), tf.exp(self.actor.logstd)
            pi = tfp.distributions.Normal(mean, std)
            # print("train_action:",action)
            # print("train_action_shape:", action.shape)  # (32,2)
            # print("~~~~~~", pi.log_prob(action))  # tf.Tensor(shape=(32, 2))
            ratio = tf.exp(pi.log_prob(action) - old_pi.log_prob(action))
            # print("ratio:", ratio)  # tf.Tensor(shape=(32,2))
            surr = ratio * adv
            if self.method == 'penalty':  # ppo penalty
                kl = tfp.distributions.kl_divergence(old_pi, pi)
                kl_mean = tf.reduce_mean(kl)
                loss = -(tf.reduce_mean(surr - self.lam * kl))
            else:  # ppo clip
                loss = -tf.reduce_mean(
                    tf.minimum(surr,
                               tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv)
                )
        a_gard = tape.gradient(loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if self.method == 'kl_pen':
            return kl_mean

    def train_actor_discrete(self, state, action, adv, old_pi):
        """
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return: kl_mean or None
        """
        with tf.GradientTape() as tape:
            all_act = self.actor(state)
            # print("all_act:", all_act)
            pi = tfp.distributions.Categorical(logits=all_act)
            # print("pi:", pi)
            # print("train_action:",action)
            # print("train_action_shape:", action.shape)  # (32,1)
            action = action.squeeze()
            # print("action.squeeze", action.shape)  # (32, )
            # print("~~~~~~", pi.log_prob(action))  # tf.Tensor(shape=(32,))
            # reshape = tf.reshape(pi.log_prob(action), [action.shape[0], -1])  # tf.Tensor(shape=(32,))
            # print("reshape:", reshape)
            # ratio = tf.exp(pi.log_prob(action) - old_pi.log_prob(action))
            ratio = tf.exp(tf.reshape(pi.log_prob(action), [action.shape[0], -1]) - tf.reshape(old_pi.log_prob(action), [action.shape[0], -1]))
            # ratio = tf.exp(pi.log_prob(all_act) - old_pi.log_prob(all_act))
            # print("ratio:", ratio)   # tf.Tensor(shape=(32,1))
            surr = ratio * adv
            if self.method == 'penalty':  # ppo penalty
                kl = tfp.distributions.kl_divergence(old_pi, pi)
                kl_mean = tf.reduce_mean(kl)
                loss = -(tf.reduce_mean(surr - self.lam * kl))
            else:  # ppo clip
                loss = -tf.reduce_mean(
                    tf.minimum(surr,
                               tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv)
                )
        a_gard = tape.gradient(loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if self.method == 'kl_pen':
            return kl_mean

    def train_critic(self, reward, state):
        """
        Update actor network
        :param reward: cumulative reward batch
        :param state: state batch
        :return: None
        """
        reward = np.array(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic(state)
            loss = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def update(self, d_action_dim):
        """
        Update parameter with the constraint of KL divergent
        :return: None
        """
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.cumulative_reward_buffer, np.float32)
        # print("a_buffer",a)

        if d_action_dim != 0:  # 离散动作
            all_act = self.actor(s)
            probs = tf.nn.softmax(all_act).numpy()
            pi = tfp.distributions.Categorical(probs)

        else:  # 连续动作
            mean, std = self.actor(s), tf.exp(self.actor.logstd)
            pi = tfp.distributions.Normal(mean, std)

        adv = r - self.critic(s)

        # update actor
        if self.method == 'kl_pen':
            for _ in range(ACTOR_UPDATE_STEPS):
                kl = self.train_actor_continuous(s, a, adv, pi)
            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            for _ in range(ACTOR_UPDATE_STEPS):
                if d_action_dim != 0:  # 离散动作
                    self.train_actor_discrete(s, a, adv, pi)
                else:
                    self.train_actor_continuous(s, a, adv, pi)

        # update critic
        for _ in range(CRITIC_UPDATE_STEPS):
            self.train_critic(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()

    def get_action(self, state, d_action_dim, greedy=False):
        """
        Choose action
        :param state: state
        :param greedy: choose action greedy or not
        :return: clipped action
        """
        state = state[np.newaxis, :].astype(np.float32)
        if d_action_dim != 0:  # 离散动作
            all_act = self.actor(state)
            # print("all_act:", all_act)  # all_act: tf.Tensor([[0. 0. 0. 0.00729172 0.00816693]], shape=(1, 5), dtype=float32)
            probs = tf.nn.softmax(all_act).numpy()
            # print("probs:", probs)  # probs: [[0.19938116 0.19938116 0.19938116 0.20084031 0.20101616]]
            if greedy:
                return np.argmax(probs.ravel())
            else:
                # choose_act = tl.rein.choice_action_by_probs(probs.ravel())
                # return_act = np.array([choose_act])
                # print("return_act:", return_act)
                pi = tfp.distributions.Categorical(probs.squeeze())
                # print("pi:", pi)  # pi: tfp.distributions.Categorical("Categorical", batch_shape=[], event_shape=[], dtype=int32)
                action = tf.squeeze(pi.sample(1), axis=0)
                # print("action", action)  # action: tf.Tensor(1, shape=(), dtype=int32)
                return np.array([action])  # [3]
        else:  # 连续动作
            mean, std = self.actor(state), tf.exp(self.actor.logstd)
            if greedy:
                action = mean[0]  # ????
            else:
                pi = tfp.distributions.Normal(mean, std)  # 离散动作时改成Catgorial分布，传入softmax的输出
                # print("pi:", pi)  # pi: tfp.distributions.Normal("Normal", batch_shape=[1, 2], event_shape=[], dtype=float32)
                action = tf.squeeze(pi.sample(1), axis=0)[0]  # choosing action
                # print("action", action)  # action tf.Tensor([-0.7383712  1.2029266], shape=(2,), dtype=float32)
            return np.clip(action, -self.action_bound, self.action_bound)  # [-1.         -0.26316592]

    def save(self, alg_name, env_id):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([alg_name, env_id]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)

    def load(self, alg_name, env_id):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([alg_name, env_id]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done:
            v_s_ = 0
        else:
            v_s_ = self.critic(np.array([next_state], np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()




