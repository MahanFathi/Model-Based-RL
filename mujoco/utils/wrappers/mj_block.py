import gym
import numpy as np

from mujoco.utils.forward import mj_forward_factory
from mujoco.utils.backward import mj_gradients_factory


class MjBlockWrapper(gym.Wrapper):
    """Wrap the forward and backward model. Further used for PyTorch blocks."""

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def gradient_factory(self, mode):
        """
        :param mode: 'dynamics' or 'reward'
        :return:
        """
        # TODO: due to dynamics and reward isolation, this isn't the
        #  most efficient way to handle this; lazy, but simple
        #env = self.clone()
        return mj_gradients_factory(self, mode)

    def forward_factory(self, mode):
        """
        :param mode: 'dynamics' or 'reward'
        :return:
        """
        #env = self.clone()
        return mj_forward_factory(self, mode)

    def gradient_wrapper(self, mode):
        """
        Decorator for making gradients be the same size the observations for example.
        :param mode: either 'dynamics' or 'reward'
        :return:
        """

        # mode agnostic for now
        def decorator(gradients_fn):
            def wrapper(*args, **kwargs):
                #if mode == "forward":
                gradients_fn(*args, **kwargs)
                #else:
                #    dfds, dfda = gradients_fn(*args, **kwargs)
                #    # no further reshaping is needed for the case of hopper, also it's mode-agnostic
                #    gradients = np.concatenate([dfds, dfda], axis=1)
                return

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
