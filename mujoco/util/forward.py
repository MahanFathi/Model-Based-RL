import numpy as np


def mj_forward_factory(env, mode):
    """
    :param env: gym.envs.mujoco.mujoco_env.mujoco_env.MujocoEnv
    :param mode: 'dynamics' or 'reward'
    :return:
    """

    @env.forward_wrapper(mode)
    def mj_forward(state_action):
        """
        :param state_action: np.array of state and action, concatenated
        :return:
        """
        qpos = state_action[:env.model.nq]
        qvel = state_action[env.model.nq:(env.model.nq + env.model.nv)]
        ctrl = state_action[-env.model.nu:]
        env.set_state(qpos, qvel)
        # set solver options for finite differences
        next_state, reward, done, _ = env.step(ctrl)
        if mode is 'dynamics':
            return np.array(next_state)
        elif mode is 'reward':
            return np.array([reward])
    return mj_forward
