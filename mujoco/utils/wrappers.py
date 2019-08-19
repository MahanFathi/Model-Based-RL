import gym
import numpy as np
from copy import deepcopy

from .forward import mj_forward_factory
from .backward import mj_gradients_factory

gym.make


class MjBlockWrapper(gym.Wrapper):
    """Wrap the forward and backward model. Further used for PyTorch blocks."""

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def clone(self):
        return deepcopy(self)

    def gradient_factory(self, mode):
        """
        :param mode: 'dynamics' or 'reward'
        :return:
        """
        # TODO: due to dynamics and reward isolation, this isn't the
        #  most efficient way to handle this; lazy, but simple
        env = self.clone()
        return mj_gradients_factory(env, mode)

    def forward_factory(self, mode):
        """
        :param mode: 'dynamics' or 'reward'
        :return:
        """
        env = self.clone()
        return mj_forward_factory(env, mode)

    def gradient_wrapper(self, mode):
        """
        Decorator for making gradients be the same size the observations for example.
        :param mode: either 'dynamics' or 'reward'
        :return:
        """

        # mode agnostic for now
        def decorator(gradients_fn):
            def wrapper(*args, **kwargs):
                dfds, dfda = gradients_fn(*args, **kwargs)
                # no further reshaping is needed for the case of hopper, also it's mode-agnostic
                gradients = np.concatenate([dfds, dfda], axis=1)
                return gradients
            return wrapper
        return decorator

    def forward_wrapper(self, mode):
        """
        Decorator for making gradients be the same size the observations for example.
        :param mode: either 'dynamics' or 'reward'
        :return:
        """

        # mode agnostic for now
        def decorator(forward_fn):
            def wrapper(*args, **kwargs):
                f = forward_fn(*args, **kwargs)  # next state
                # no further reshaping is needed for the case of hopper, also it's mode-agnostic
                return f
            return wrapper
        return decorator

    @staticmethod
    def is_done(state):
        height, ang = state[1:3]
        done = not (np.isfinite(state).all() and (np.abs(state[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        return done


class RewardScaler(gym.RewardWrapper):
    """Bring rewards to a reasonable scale."""

    def __init__(self, env, scale=0.01):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale
