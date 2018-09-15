import sys
import gym
from gym.spaces import *

def print_spaces(space):
    print(space)
    if isinstance(space, Box):
        print('SPACE LOW: ' + str(space.low))
        print('SPACE HIGH: ' + str(space.high))

if __name__ == '__main__':
    env = gym.make(sys.argv[1])

    print('[Observation Space]')
    print_spaces(env.observation_space)
    print()

    print('[Action Space]')
    print_spaces(env.action_space)
    print()

    try:
        print('ACTION DESCRIPTION: ' + str(env.unwrapped.get_action_meanings()))
    except AttributeError:
        pass
