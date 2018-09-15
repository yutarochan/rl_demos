#!/usr/bin/env python
import gym

# Create Mountain Car Environment
env = gym.make('HandManipulateBlock-v0')
env.reset()

# Execute Environment
for i in range(5000):
    env.render()
    env.step(env.action_space.sample())
