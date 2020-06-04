import numpy as np
from gym import spaces
import gym

class DiscreteActionWrapper(gym.Wrapper):
    """ 
    Discretized the continuous action for CarRacing-v0
    Ref: https://notanymike.github.io/Solving-CarRacing/
    """
    def __init__(self, env):
        super(DiscreteActionWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(5)
        # switch order for observation space
        dim1, dim2, channel = env.observation_space.shape  
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf, shape=(channel, dim1, dim2))

    def step(self, action):
        if action == 0:  # Turn_left
            continuous_action = [ -1.0, 0.0, 0.0 ]
        elif action == 1: # Turn_right
            continuous_action = [ +1.0, 0.0, 0.0 ]
        elif action == 2: # Brake
            continuous_action = [ 0.0, 0.0, 1.0 ]
        elif action == 3: # Accelerate
            continuous_action = [ 0.0, 1.0, 0.0 ]
        elif action == 4: # Do-Nothing
            continuous_action = [ 0.0, 0.0, 0.0 ]
        else:
            raise ValueError
    
        observation, reward, done, info = self.env.step(continuous_action)
        return np.moveaxis(observation, 2, 0), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return np.moveaxis(observation, 2, 0)  # (H, W, C) -> (C, H, W)

    def render(self, **kwargs):
        pass


class ObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)
        # switch order for observation space
        dim1, dim2, channel = env.observation_space.shape  
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf, shape=(channel, dim1, dim2))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return np.moveaxis(observation, 2, 0), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return np.moveaxis(observation, 2, 0)  # (H, W, C) -> (C, H, W)     


if __name__ == '__main__':
    # test
    EnvName = 'CarRacing-v0'
    # env = DiscreteActionWrapper(gym.make(EnvName))
    env =gym.make(EnvName)

    env.reset()
    for _ in range(10000):
        # env.render()
        a = env.action_space.sample()
        print(a)
        env.step(a) # take a random action
    env.close()


