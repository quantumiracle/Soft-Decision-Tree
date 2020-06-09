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
    def __init__(self, env, channel=None):
        """  
        Transform the observation for Atari games: downsample, select channel, truncate the edge pixels

        channel: selected channel index if not None
        
        """
        super(ObservationWrapper, self).__init__(env)
        # switch order for observation space for using PyTorch model
        dim1, dim2, channel = env.observation_space.shape  
        self.downsample_rate=3
        self.channel = channel
        self.dim1 = int((dim1-30)/self.downsample_rate)
        self.dim2 = int((dim2-10)/self.downsample_rate)
        if isinstance(self.channel, int):
            self.observation_space = spaces.Box(low=-np.inf,high=np.inf, shape=(1, self.dim1, self.dim2))
        elif self.channel is None:
            self.observation_space = spaces.Box(low=-np.inf,high=np.inf, shape=(1, self.dim1, self.dim2))
        else:
            raise NotImplementedError
        print(self.observation_space)
    
    def prepro(self, I):
        """Downsample 210x160x3 uint8 frame into 95x80x3."""
        if isinstance(self.channel, int):
            I=I[10:190, 10:, self.channel]
        elif self.channel is None:
            I=I[10:190, 10:]
        else:
            raise NotImplementedError

        I = I[::self.downsample_rate, ::self.downsample_rate]
        return I

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # return np.moveaxis(self.prepro(observation), 2, 0), reward, done, info
        return self.prepro(observation), reward, done, info


    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        # return np.moveaxis(self.prepro(observation), 2, 0)  # (H, W, C) -> (C, H, W)     
        return self.prepro(observation)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test
    # EnvName = 'CarRacing-v0'
    # env = DiscreteActionWrapper(gym.make(EnvName))
    # env =gym.make(EnvName)

    EnvName = 'Freeway-v0'
    env = ObservationWrapper(gym.make(EnvName))

    env.reset()
    for _ in range(10000):
        # env.render()
        a = env.action_space.sample()
        print(a)
        s, r, d, _ = env.step(a) # take a random action
        # plt.imshow(np.moveaxis(s, 0, 2))
        plt.imshow(s)
        plt.show()
    env.close()


