import unittest

import torch
import numpy as np
from model.config import get_cfg_defaults
from mujoco import build_agent
from copy import deepcopy


class TestGradients(unittest.TestCase):

    def test_reward_gradients(self):
        cfg = get_cfg_defaults()
        cfg.MUJOCO.ENV = "InvertedPendulumEnv"
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
        cfg.merge_from_file("/home/aleksi/Workspace/Model-Based-RL/configs/swimmer.yaml")
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

    def test_combined_gradients(self):

        # Run multiple times with different sigma for generating action values
        N = 100
        sigma = np.linspace(1e-2, 1e1, N)
        cfg_file = "/home/aleksi/Workspace/Model-Based-RL/configs/swimmer.yaml"

        for s in sigma:
            self.run_combined_gradients(cfg_file, s)

    def run_combined_gradients(self, cfg_file, sigma):

        cfg = get_cfg_defaults()
        cfg.merge_from_file(cfg_file)
        agent = build_agent(cfg)
        mj_forward_fn = agent.forward_factory("forward")
        mj_gradients_fn = agent.gradient_factory("forward")

        # Start from the same state with constant action, make sure reward is equal in both repetitions

        # Drive both simulations forward 5 steps
        nwarmup = 5

        # Reset and get initial state
        agent.reset()
        init_qpos = agent.data.qpos.copy()
        init_qvel = agent.data.qvel.copy()

        # Set constant action
        na = agent.model.actuator_acc0.shape[0]
        action = torch.DoubleTensor(np.random.randn(na)*sigma)

        # Do first simulation
        for _ in range(nwarmup):
            r = mj_forward_fn(action)

        # Take a snapshot of this state so we can use it in gradient calculations
        agent.data.ctrl[:] = action.detach().numpy().copy()
        data = agent.get_snapshot()

        # Calculate reward for the next step
        reward1 = mj_forward_fn(action)

        # Reset and set to initial state, then do the second simulation; this time call mj_forward_fn without args
        agent.reset()
        agent.data.qpos[:] = init_qpos
        agent.data.qvel[:] = init_qvel
        for _ in range(nwarmup):
            agent.data.ctrl[:] = action.detach().numpy()
            r = mj_forward_fn()

        # Get reward for next step
        agent.data.ctrl[:] = action.detach().numpy()
        reward2 = mj_forward_fn()

        # reward1 and reward2 should be equal
        self.assertEqual(reward1, reward2, "Simulations from same initial state diverged")

        # Then make sure simulation from snapshot doesn't diverge from original simulation
        agent.set_snapshot(data)
        reward_snapshot = mj_forward_fn()
        self.assertEqual(reward1, reward_snapshot, "Simulation from snapshot diverged")

        # Make sure simulations are correct in the gradient calculations as well
        dsdsa = mj_gradients_fn(data, reward1, test=True)


if __name__ == '__main__':
    unittest.main()
