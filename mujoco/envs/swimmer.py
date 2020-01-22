import os
import torch
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    COPIED FROM GYM. W/ SLIGHT MODIFICATIONS:
        * READING FROM OWN .XML.
        * FULL STATE OBSERVATIONS, I.E. QPOS CONCAT'D WITH QVEL.
        * is_done METHOD SHOULD BE IMPLEMENTED
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.frame_skip = self.cfg.MODEL.FRAME_SKIP
        mujoco_assets_dir = os.path.abspath("./mujoco/assets/")
        mujoco_env.MujocoEnv.__init__(self, os.path.join(mujoco_assets_dir, "swimmer.xml"), self.frame_skip)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd# + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        """DIFFERENT FROM ORIGINAL GYM"""
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        self.sim.reset()
        if self.cfg.MODEL.POLICY.NETWORK:
            self.set_state(
                self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
                self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
            )
        else:
            self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    @staticmethod
    def is_done(state):
        done = False
        return done

    def tensor_reward(self, state, action, next_state):
        """DIFFERENT FROM ORIGINAL GYM"""
        ctrl_cost_coeff = 0.0001
        xposbefore = state[0]
        xposafter = next_state[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * torch.sum(torch.mul(action, action))
        reward = reward_fwd + reward_ctrl
        return reward.view([1, ])
