import os
import torch
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import math


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    COPIED FROM GYM. W/ SLIGHT MODIFICATIONS:
        * READING FROM OWN .XML.
        * FULL STATE OBSERVATIONS, I.E. QPOS CONCAT'D WITH QVEL.
        * is_done METHOD SHOULD BE IMPLEMENTED
        * torch implementation of reward function
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.frame_skip = self.cfg.MODEL.FRAME_SKIP
        mujoco_assets_dir = os.path.abspath("./mujoco/assets/")
        self.initialised = False
        mujoco_env.MujocoEnv.__init__(self, os.path.join(mujoco_assets_dir, "hopper.xml"), self.frame_skip)
        utils.EzPickle.__init__(self)
        self.initialised = True

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        coeff = min(max(height/1.25, 0), 1)*0.5 + max(((math.pi - abs(ang))/math.pi), 1)*0.5
        reward += coeff * alive_bonus
        #reward += alive_bonus
        #reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2)) and self.initialised
        #done = not (np.isfinite(s).all().all() and
        #            (height > .7)) and self.initialised
        ob = self._get_obs()
        return ob, reward, False, {}

    def _get_obs(self):
        """DIFFERENT FROM ORIGINAL GYM"""
        return np.concatenate([
            self.sim.data.qpos.flat,  # this part different from gym. expose the whole thing.
            self.sim.data.qvel.flat,  # this part different from gym. clip nothing.
        ])

    def reset_model(self):
        self.sim.reset()
        #qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        #qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        #self.set_state(qpos, qvel)
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    @staticmethod
    def is_done(state):
        height, ang = state[1:3]
        done = not (np.isfinite(state).all() and (np.abs(state[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        return done

    def tensor_reward(self, state, action, next_state):
        """DIFFERENT FROM ORIGINAL GYM"""
        posbefore = state[0]
        posafter, height, ang = next_state[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * torch.sum(torch.mul(action, action))
        return reward.view([1, ])
