import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from mujoco.utils import mj_gradients_factory
from mujoco.utils import mj_forward_factory


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, cfg):
        self.cfg = cfg
        mujoco_assets_dir = os.path.abspath(self.cfg.MUJOCO.ASSETS_PATH)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(mujoco_assets_dir, "hopper.xml"), 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,  # this part different from gym. expose the whole thing.
            self.sim.data.qvel.flat,  # this part different from gym. clip nothing.
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    @classmethod
    def gradient_factory(cls, cfg, mode):
        """
        :param cfg: root config node
        :param mode: 'dynamics' or 'reward'
        :return:
        """
        # TODO: due to dynamics and reward isolation, this isn't the most efficient way to handle this,
        #   lazy, but simple
        env = cls(cfg)
        return mj_gradients_factory(env, mode)

    @classmethod
    def forward_factory(cls, cfg, mode):
        """
        :param cfg: root config node
        :param mode: 'dynamics' or 'reward'
        :return:
        """
        env = cls(cfg)
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
