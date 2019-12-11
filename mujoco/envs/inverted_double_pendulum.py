import os
import torch
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class InvertedDoublePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    COPIED FROM GYM. W/ SLIGHT MODIFICATIONS:
        * READING FROM OWN .XML.
        * FULL STATE OBSERVATIONS, I.E. QPOS CONCAT'D WITH QVEL.
        * is_done METHOD SHOULD BE IMPLEMENTED
        * torch implementation of reward function
    """

    def __init__(self):
        mujoco_assets_dir = os.path.abspath("./mujoco/assets/")
        mujoco_env.MujocoEnv.__init__(self, os.path.join(mujoco_assets_dir, "inverted_double_pendulum.xml"), 4)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        r = - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, done, {}

    def _get_obs(self):
        """DIFFERENT FROM ORIGINAL GYM"""
        return np.concatenate([
            self.sim.data.qpos.flat,  # this part different from gym. expose the whole thing.
            self.sim.data.qvel.flat,  # this part different from gym. clip nothing.
        ])

    def reset_model(self):
        self.set_state(
            self.init_qpos,# + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel# + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]

    @staticmethod
    def is_done(state):
        arm_length = 0.6
        theta_1 = state[1]
        theta_2 = state[2]
        y = arm_length * np.cos(theta_1) + \
            arm_length * np.sin(theta_1 + theta_2)
        done = bool(y <= 1)
        return done

    def tensor_reward(self, state, action, next_state):
        """DIFFERENT FROM ORIGINAL GYM"""
        arm_length = 0.6
        theta_1 = next_state[1]
        theta_2 = next_state[2]
        y = arm_length * torch.cos(theta_1) + \
            arm_length * torch.cos(theta_1 + theta_2)
        x = arm_length * torch.cos(theta_1) + \
            arm_length * torch.cos(theta_1 + theta_2) + \
            next_state[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = next_state[4:6]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        reward = alive_bonus - dist_penalty - vel_penalty
        return reward.view([1, ])
