import unittest

import torch
import numpy as np
from model.config import get_cfg_defaults
from mujoco import build_agent


class TestGradients(unittest.TestCase):

    def test_reward_gradients(self):
        cfg = get_cfg_defaults()
        cfg.MUJOCO.ENV = "HopperEnv"
        agent = build_agent(cfg)
        mj_reward_forward_fn = agent.forward_factory("reward")
        mj_reward_gradients_fn = agent.gradient_factory("reward")

        nwarmup = 5
        agent.reset()
        nv = agent.sim.model.nv
        nu = agent.sim.model.nu
        for _ in range(nwarmup):
            action = torch.Tensor(agent.action_space.sample())
            ob, r, _, _ = agent.step(action)
        state = ob.detach().numpy()
        action = agent.action_space.sample()
        state_action = np.concatenate([state, action], axis=0)
        drdsa = mj_reward_gradients_fn(state_action)
        drds = drdsa[:, :nv * 2]
        drda = drdsa[:, -nu:]

        eps = 1e-6
        state_action_prime = state_action + eps
        r = mj_reward_forward_fn(state_action)
        r_prime = mj_reward_forward_fn(state_action_prime)

        r_prime_estimate = r + \
                           np.squeeze(np.matmul(drds, np.array([eps] * 2 * nv).reshape([-1, 1]))) + \
                           np.squeeze(np.matmul(drda, np.array([eps] * nu).reshape([-1, 1])))
        self.assertAlmostEqual(r_prime[0], r_prime_estimate[0], places=5)

    def test_dynamics_gradients(self):
        cfg = get_cfg_defaults()
        cfg.MUJOCO.ENV = "HopperEnv"
        agent = build_agent(cfg)
        mj_dynamics_forward_fn = agent.forward_factory("dynamics")
        mj_dynamics_gradients_fn = agent.gradient_factory("dynamics")

        nwarmup = 5
        agent.reset()
        nv = agent.sim.model.nv
        nu = agent.sim.model.nu
        for _ in range(nwarmup):
            action = torch.Tensor(agent.action_space.sample())
            ob, r, _, _ = agent.step(action)
        state = ob.detach().numpy()
        action = agent.action_space.sample()
        state_action = np.concatenate([state, action], axis=0)
        dsdsa = mj_dynamics_gradients_fn(state_action)
        dsds = dsdsa[:, :nv * 2]
        dsda = dsdsa[:, -nu:]

        eps = 1e-6
        state_action_prime = state_action + eps
        s = mj_dynamics_forward_fn(state_action)
        s_prime = mj_dynamics_forward_fn(state_action_prime)

        s_prime_estimate = s + \
                           np.squeeze(np.matmul(dsds, np.array([eps] * 2 * nv).reshape([-1, 1]))) + \
                           np.squeeze(np.matmul(dsda, np.array([eps] * nu).reshape([-1, 1])))
        print(s)
        print(s_prime)
        print(s_prime_estimate)
        self.assert_(np.allclose(s_prime, s_prime_estimate))


if __name__ == '__main__':
    unittest.main()
