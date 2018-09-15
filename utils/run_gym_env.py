import sys
import gym

def run_gym_env(argv):
    # Create Environment Based on CLI Argument
    env = gym.make(argv[1])
    obv = env.reset()

    # Execute Environment
    for i in range(int(argv[2])):
        env.render()
        env.step(env.action_space.sample())

    env.close()

if __name__ == '__main__':
    run_gym_env(sys.argv)
