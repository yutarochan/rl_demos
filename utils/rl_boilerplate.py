'''
OpenAI Gym RL Boilerplate Code
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import sys
import gym

# Application Parameters
MAX_EP_NUM = 150
MAX_EP_STEP = 5000

def run_env(env):
    for ep in range(MAX_EP_NUM):
        obv = env.reset()   # Reset Environment

        for st in range(MAX_EP_STEP):
            env.render()

            action = env.action_space.sample() # Replace with Agent
            next_state, reward, done, info = env.step(action)

            obs = next_state

            if done:
                print('EPISODE #' + str(ep+1) + ' ENDED IN ' + str(st) + ' STEPS.')
                break

if __name__ == '__main__':
    env = gym.make("VideoPinball-v0")
    run_env(env)
    env.close()
