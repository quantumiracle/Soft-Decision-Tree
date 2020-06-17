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
    def __init__(self, env, selected_channel=None):
        """  
        Transform the observation for Atari games: downsample, select channel, truncate the edge pixels

        selected_channel: selected channel index if not None; also can be 'grey' to transform into grey-scale image
        """
        super(ObservationWrapper, self).__init__(env)
        # switch order for observation space for using PyTorch model
        dim1, dim2, channel = env.observation_space.shape  
        self.downsample_rate=3
        self.selected_channel = selected_channel
        self.dim1 = int((dim1-30)/self.downsample_rate)
        self.dim2 = int((dim2-10)/self.downsample_rate)
        if isinstance(self.selected_channel, int) or self.selected_channel == 'grey':
            self.observation_space = spaces.Box(low=-np.inf,high=np.inf, shape=(1, self.dim1, self.dim2))
        elif self.selected_channel is None:
            self.observation_space = spaces.Box(low=-np.inf,high=np.inf, shape=(channel, self.dim1, self.dim2))
        else:
            raise NotImplementedError
    
    def prepro(self, I):
        """Downsample 210x160x3 uint8 frame into 95x80x3."""
        cropped_I = I[10:190, 10:]
        if isinstance(self.selected_channel, int):
            I=cropped_I[:, :, self.selected_channel:self.selected_channel+1]  # preserve the third dimension
        elif self.selected_channel is 'grey':
            # transform RGB image to grey-scale image, following: https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert
            I = np.expand_dims(np.dot(cropped_I[...,:3], [0.2989, 0.5870, 0.1140]), 2)
        else:
            I = cropped_I

        I = I[::self.downsample_rate, ::self.downsample_rate]
        return I

    def step(self, action):
        # normalize the value and swap axis
        observation, reward, done, info = self.env.step(action)
        return np.moveaxis(self.prepro(observation), 2, 0)/255., reward, done, info



    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return np.moveaxis(self.prepro(observation), 2, 0)/255.  # (H, W, C) -> (C, H, W)     

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # test
    # EnvName = 'CarRacing-v0'
    # env = DiscreteActionWrapper(gym.make(EnvName))
    # env =gym.make(EnvName)

    EnvName = 'Freeway-v0'
    env = ObservationWrapper(gym.make(EnvName), selected_channel='grey')

    env.reset()
    for _ in range(10000):
        # env.render()
        a = env.action_space.sample()
        print(a)
        s, r, d, _ = env.step(a) # take a random action
        print(s.shape)
        # plt.imshow(np.moveaxis(s, 0, 2))
        # plt.imshow(np.moveaxis(s, 0, 2)[:, :, 0], vmin=0, vmax=1)  # when selected channel is int
        plt.imshow(np.moveaxis(s, 0, 2)[:, :, 0], cmap=plt.get_cmap('gray'), vmin=0, vmax=1)  # when selected channel is 'grey'
        plt.show()
    env.close()


