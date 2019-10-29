import numpy as np
import torch

def mj_forward_factory(agent, mode):
    """
    :param env: gym.envs.mujoco.mujoco_env.mujoco_env.MujocoEnv
    :param mode: 'dynamics' or 'reward'
    :return:
    """

    @agent.forward_wrapper(mode)
    def mj_forward(action=None):
        """
        :param state_action: np.array of state and action, concatenated
        :return:
        """
        #qpos = state_action[:env.model.nq]
        #qvel = state_action[env.model.nq:(env.model.nq + env.model.nv)]
        #ctrl = state_action[-env.model.nu:]
        #env.set_state(qpos, qvel)
        # set solver options for finite differences

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
        next_state, reward, done, _ = agent.step(action)

        if mode == "dynamics":
            return np.array(next_state)
        elif mode in ["reward", "forward"]:
            return np.array([reward])
    return mj_forward
