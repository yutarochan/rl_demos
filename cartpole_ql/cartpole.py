'''
Q-Learning Cartpole
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import sys
import gym
import numpy as np
from QL import QLearner

# Application Parameters
MAX_EP_NUM = 50000

def train(agent, env):
    best_reward = -float('inf')
    for ep in range(MAX_EP_NUM):
        done = False
        obs = env.reset()
        total_reward = 0.0

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward

        if total_reward > best_reward:
            best_reward = total_reward

        print('EPISODE #' + str(ep) + ' WITH REWARD ' + str(total_reward) + ' [BEST REWARD = ' + str(best_reward) + '] EPS: ' + str(agent.epsilon))

    # Return Best Q Policy
    return np.argmax(agent.Q, axis=4)

def test(agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0

    while not done:
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward

    return total_reward

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = QLearner(env)
    agent.print_params()

    policy = train(agent, env)

    gym_monitor_path = './gym_monitor_output'
    env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)

    for i in range(1000): test(agent, env, policy)

    env.close()
