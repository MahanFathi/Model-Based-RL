import os
import torch
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    COPIED FROM GYM. W/ SLIGHT MODIFICATIONS:
        * READING FROM OWN .XML.
        * FULL STATE OBSERVATIONS, I.E. QPOS CONCAT'D WITH QVEL.
        * is_done METHOD SHOULD BE IMPLEMENTED
        * torch implementation of reward function
    """

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_assets_dir = os.path.abspath("./mujoco/assets/")
        mujoco_env.MujocoEnv.__init__(self, os.path.join(mujoco_assets_dir, "inverted_pendulum.xml"), 2)

    def step(self, a):
        """DIFFERENT FROM ORIGINAL GYM"""
        arm_length = 0.6
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        theta = ob[1]
        y = arm_length * np.cos(theta)
        x = arm_length * np.cos(theta)
        dist_penalty = 0.01 * x ** 2 + (y - 1) ** 2
        v = ob[3]
        vel_penalty = 1e-3 * v ** 2
        reward = -dist_penalty - vel_penalty
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    @staticmethod
    def is_done(state):
        done = False
        return done

    def tensor_reward(self, state, action, next_state):
        """DIFFERENT FROM ORIGINAL GYM"""
        arm_length = 0.6
        theta = next_state[1]
        y = arm_length * torch.cos(theta)
        x = arm_length * torch.cos(theta)
        dist_penalty = 0.01 * x ** 2 + (y - 1) ** 2
        v = next_state[3]
        vel_penalty = 1e-3 * v ** 2
        reward = -dist_penalty - vel_penalty
        return reward.view([1, ])
