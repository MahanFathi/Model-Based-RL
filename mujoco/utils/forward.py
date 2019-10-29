import numpy as np


def mj_forward_factory(agent, mode):
    """
    :param env: gym.envs.mujoco.mujoco_env.mujoco_env.MujocoEnv
    :param mode: 'dynamics' or 'reward'
    :return:
    """

    @agent.forward_wrapper(mode)
    def mj_forward(action):
        """
        :param state_action: np.array of state and action, concatenated
        :return:
        """
        #qpos = state_action[:env.model.nq]
        #qvel = state_action[env.model.nq:(env.model.nq + env.model.nv)]
        #ctrl = state_action[-env.model.nu:]
        #env.set_state(qpos, qvel)
        # set solver options for finite differences
        next_state, reward, done, _ = agent.step(action)
        if mode == "dynamics":
            return np.array(next_state)
        elif mode in ["reward", "forward"]:
            return np.array([reward])
    return mj_forward
