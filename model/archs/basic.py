import torch
import torch.nn as nn
from model.blocks import build_policy, mj_torch_block_factory
from copy import deepcopy
import numpy as np


class Basic(nn.Module):
    def __init__(self, cfg, agent):
        """Build the model from the fed config node.
        :param cfg: CfgNode containing the configurations of everything.
        """
        super(Basic, self).__init__()
        self.cfg = cfg
        self.agent = agent

        # build policy net
        self.policy_net = build_policy(cfg.MODEL.POLICY, self.agent)

        # build forward dynamics block
        #self.dynamics_block = mj_torch_block_factory(agent, 'dynamics').apply
        #self.reward_block = mj_torch_block_factory(agent, 'reward').apply
        self.forward_block = mj_torch_block_factory(agent, 'forward').apply

    def forward(self, state):
        """Single pass.
        :param state:
        :return:
        """

        # get action
        action = self.policy_net(state)
        #if not self.training:
        #    return action
        #state_action = torch.cat([state, action])
        #self.agent.env.sim.data.ctrl[:] = action
        #next_state = self.dynamics_block(state_action)

        #snapshot = self.agent.get_snapshot()

        #next_state = self.dynamics_block(action)
        #reward = self.reward_block(state_action)

        #self.agent.set_snapshot(snapshot)
        #reward = self.reward_block(action)
        #reward = self.agent.tensor_reward(state, action, next_state)

        # Forward block will drive the simulation forward and return reward as a tensor
        reward = self.forward_block(action)

        # We can grab the next state from the simulation
        next_state = torch.Tensor(np.concatenate((self.agent.env.sim.data.qpos, self.agent.env.sim.data.qvel)))

        return next_state, reward
