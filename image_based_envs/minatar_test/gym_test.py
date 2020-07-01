from minatar import Environment
from gym_wrapper import GymWrapper, LowDimWrapper
import argparse
import numpy as np

GameName='freeway'

# env = GymWrapper(Environment(GameName))
env = LowDimWrapper(Environment(GameName))
print(env.observation_space)

state = env.reset()
print('Observation Space: {}   Action Space: {}'.format(env.observation_space, env.action_space))
for step in range(10):
    # env.render()
    next_state, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print(step, reward)
#     np.save('state/{}'.format(step), state)
    state = next_state
    print(state.shape)
    if done:
        env.reset()
env.close()