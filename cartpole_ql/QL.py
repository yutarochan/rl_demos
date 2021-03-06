'''
Q-Learning Algorithm
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
import numpy as np

# Application Hyperparameters
EPSILON_MIN = 0.01
MAX_NUM_EP = 50000
STEPS_PER_EP = 200
ALPHA = 0.045
GAMMA = 0.9975
NUM_DISC_BIN = 100
MAX_STEPS = MAX_NUM_EP * STEPS_PER_EP
EPSILON_DECAY = 500 * EPSILON_MIN / MAX_STEPS

class QLearner:
    def __init__(self, env):
        # Initialize Parameters
        '''
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins = NUM_DISC_BIN
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
        '''

        # Manually Define Parameters
        self.obs_shape = 4
        self.obs_high = np.array([env.observation_space.high[0], 100, env.observation_space.high[2], 100])
        self.obs_low = np.array([env.observation_space.low[0], -100, env.observation_space.low[2], -100])
        self.obs_bins = NUM_DISC_BIN
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n

        # Define Q-Values
        self.Q = np.zeros((self.obs_bins + 1, self.obs_bins + 1, self.obs_bins + 1, self.obs_bins + 1, self.action_shape))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0

    def print_params(self):
        print('OBS SHAPE:\t' + str(self.obs_shape))
        print('OBS HIGH:\t' + str(self.obs_high))
        print('OBS LOW:\t' + str(self.obs_low))
        print('OBS BINS:\t' + str(self.obs_bins))
        print('OBS WIDTH:\t' + str(self.bin_width))
        print('Q SHAPE:\t' + str(self.Q.shape))
        print('ALPHA:\t\t' + str(self.alpha))
        print('GAMMA:\t\t' + str(self.gamma))
        print('EPSILON:\t' + str(self.epsilon))

    def discretize(self, obs):
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        disc_obs = self.discretize(obs)

        # Epsilon Greedy Acion Selection
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[disc_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, next_obs):
        disc_obs = self.discretize(obs)
        disc_nxt_obs = self.discretize(next_obs)

        td_target = reward + self.gamma * np.max(self.Q[disc_nxt_obs])
        td_error = td_target - self.Q[disc_obs][action]
        self.Q[disc_obs][action] += self.alpha * td_error
