import numpy as np
import torch


def mj_forward_factory(agent, mode):
    """
    :param agent: gym.envs.mujoco.mujoco_env.mujoco_env.MujocoEnv
    :param mode: 'dynamics' or 'reward'
    :return:
    """

    @agent.forward_wrapper(mode)
    def mj_forward(action=None):
        """
        :param action: np.array of action -- if missing, agent.data.ctrl is used as action
        :return:
        """

        # If action wasn't explicitly given use the current one in agent's env
        if action is None:
            action = agent.data.ctrl

        # Convert tensor to numpy array, and make sure we're using an action value that isn't referencing agent's data
        # (otherwise we might get into trouble if ctrl is limited and frame_skip is larger than one)
        if isinstance(action, torch.Tensor):
            action = action.detach().numpy().copy()
        elif isinstance(action, np.ndarray):
            action = action.copy()
        else:
            raise TypeError("Expecting a torch tensor or numpy ndarray")

        # Make sure dtype is numpy.float64
        assert action.dtype == np.float64, "You must use dtype numpy.float64 for actions"

        # Advance simulation with one step
        next_state, agent.reward, agent.is_done, _ = agent.step(action)

        return next_state

    return mj_forward
