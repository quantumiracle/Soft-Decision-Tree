import numpy as np
from gym import spaces


class GymWrapper():
    ''' Openai Gym wrapper for MinAtar '''
    env = None

    # Set this in SOME subclasses
    metadata = {'render.modes': ['human', 'rgb_a']}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, env):
        """
        Initializes the Gym wrapper.
        Args:
            env (MujocoEnv instance): The environment to wrap.
            keys (list of strings): If provided, each observation will
                consist of concatenated keys from the wrapped environment's
                observation dictionary. Defaults to robot-state and object-state.
        """
        self.env = env
        self.render_on=False
        # set up observation space
        high = np.inf  # need to specify
        low = -high

        self.observation_space = spaces.Box(low=low,high=high, shape=env.state_shape())

        # setup action space
        self.action_space = spaces.Discrete(n=env.num_actions())

    def step(self, action):
        reward, done = self.env.act(action)
        next_state = self.env.state()
        info = None
        return np.moveaxis(next_state, 2, 0), reward, done, info

    def reset(self):
        self.env.reset()
        state = self.env.state()
        return np.moveaxis(state, 2, 0)  # (H, W, C) -> (C, H, W)

    def render(self, display_interval=10):
        self.render_on = True
        self.env.display_state(display_interval)

    def render_off(self):
        self.env.close_display()
        self.render_on = False

    def close(self):
        if self.render_on:
            self.env.close_display()

    def seed(self, seed):
        pass

    @property
    def unwrapped(self):
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        # propagate exception
        return False


class LowDimWrapper(GymWrapper):
    """ Change high dimension state (n,10,10) into low dimensional ones (9*2) """
    def __init__(self, env):
        super(LowDimWrapper, self).__init__(env)
        high = np.inf  # need to specify
        low = -high
        self.observation_space = spaces.Box(low=low,high=high, shape=(18,))

    def get_low_dimension_state(self, img):
        agent_channel = 0 # the channel of state indicating the agent in MinAtar freeway 
        cars_channel = 1 # the channel of state indicating other cars in MinAtar freeway 
        agent_pos = np.concatenate(np.where(img[0]))  # get agent coordinates from agent channel
        cars_pos = np.vstack(np.where(img[1])).T
        low_state = np.array([agent_pos.tolist()]+cars_pos.tolist()) # first is agent position, the rest are cars position
        return low_state

    def step(self, action):
        high_dim_state, reward, done, info = super(LowDimWrapper, self).step(action)
        low_dim_state = self.get_low_dimension_state(high_dim_state)
        return low_dim_state.reshape(-1), reward, done, info

    def reset(self):
        high_dim_state = super(LowDimWrapper, self).reset()
        low_dim_state = self.get_low_dimension_state(high_dim_state)
        return low_dim_state.reshape(-1)

