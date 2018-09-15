'''
Cartpole with Deep Q-Learning + Experience Replay
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import sys
import gym
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from SLP import SLP
from decay_schedule import LinearDecaySchedule
from expmem import ExperienceMemory, Experience

# Application Hyperparmeters
MAX_NUM_EP = 100000
MAX_STEPS_PER_EP = 300
REPLAY_BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Shallow Q-Learner
class ShallowQL(object):
    def __init__(self, state_shape, action_shape, lr=0.005, gamma=0.98):
        # Parameters
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.lr = lr

        # Neural Network Params
        self.Q = SLP(state_shape, action_shape)
        self.Q_opt = torch.optim.Adam(self.Q.parameters(), lr=1e-3)

        # Epsilon-Greedy Policy
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(self.epsilon_max, self.epsilon_min, 0.5 * MAX_NUM_EP * MAX_STEPS_PER_EP)

        self.step_num = 0

    def get_action(self, observation):
        return self.policy(observation)

    ''' OLD METHOD '''
    def epsilon_greedy_Q(self, observation):
        # Decay Epsilon/Exploration per Schedule
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([i for i in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(observation).data.numpy())
        return action

    def learn (self, s, a, r, s_nxt):
        td_target = r + self.gamma * torch.max(self.Q(s_nxt))
        td_error = F.mse_loss(self.Q(s)[a], td_target)

        # Update Q Estimate
        self.Q_opt.zero_grad()
        td_error.backward()
        self.Q_opt.step()

    ''' Experience Replay Method '''
    def replay_experience(self, batch_size):
        exp_batch = self.memory.sample(batch_size)
        self.learn_batch_exp(exp_batch)

    # Learning Method of Mini-Batch Experience
    def learn_batch_exp(self, experieces):
        batch_exp = Experience(*zip(*experieces))

        # Convert to Numpy Tyoe
        obs_batch = np.array(batch_exp.obs)
        action_batch = np.array(batch_exp.action)
        reward_batch = np.array(batch_exp.reward)
        next_obs_batch = np.array(batch_exp.next_obs)
        done_batch = np.array(batch_exp.done)

        # Compute Target wrt Next Observational Batch
        # Note: Multiply with Non-Done Batches (= Succesful Batches, all others are failures)
        td_target = reward_batch + ~done_batch * np.tile(self.gamma, len(next_obs_batch)) * self.Q(next_obs_batch).detach().max(1)[0].data
        td_target = td_target.to(device)

        # Compute Temporal Difference Error
        action_idx = torch.from_numpy(action_batch).to(device)
        td_error = F.mse_loss(self.Q(obs_batch).gather(1, action_idx.view(-1, 1)), td_target.float().unsqueeze(1))

        self.Q_opt.zero_grad()
        td_error.mean().backward() # Why Take the Mean?
        self.Q_opt.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.n

    agent = ShallowQL(obs_shape, action_shape)

    first_ep = True
    episode_reward = []

    for ep in range(MAX_NUM_EP):
        obs = env.reset()
        cum_reward = 0.0

        for step in range(MAX_STEPS_PER_EP):
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)

            obs = next_obs
            cum_reward += reward

            if done:
                if first_ep:
                    max_reward = cum_reward
                    first_ep = False
                episode_reward.append(cum_reward)
                if cum_reward > max_reward:
                    max_reward = cum_reward
                print('\nEPISODE #{} ENDED IN {} STEPS.\tREWARD: {}\tMEAN REWARD: {}\t BEST REWARD: {}'.format(ep, step + 1, cum_reward, np.mean(episode_reward), max_reward))
                break

    env.close()
