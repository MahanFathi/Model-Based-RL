import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from mujoco.util.gradientor import mj_gradients_factory


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def reward_calculator(self, ):
        pass

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_mj_state(self):
        return self.sim.get_state()

    def gradient_wrapper(self, mode):
        """
        Decorator for making gradients be the same size the observations for example.
        :param mode: either 'dynamics' or 'reward'
        :return:
        """
        if mode is 'dynamics':
            def decorator(gradients_fn):
                def wrapper(*args, **kwargs):
                    dfds, dfda = gradients_fn(*args, **kwargs)
                return wrapper
            return decorator
        elif mode is 'reward':
            pass
        else:
            raise Exception('give me a valid mode')

    @classmethod
    def gradient_factory(cls, mode):
        """
        :param mode: 'dynamics' or 'reward'
        :return:
        """
        # TODO: due to dynamics and reward isolation, this isn't the most efficient way to handle this,
        #   lazy, but simple
        env = cls()
        return mj_gradients_factory(env, mode)









