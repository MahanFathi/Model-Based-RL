import torch
import torch.nn as nn
from model.blocks import build_policy, mj_torch_block_factory
from copy import deepcopy
from solver import build_optimizer
import numpy as np
from ..blocks.policy.strategies import *


class Basic(nn.Module):
    def __init__(self, cfg, agent):
        """Build the model from the fed config node.
        :param cfg: CfgNode containing the configurations of everything.
        """
        super(Basic, self).__init__()
        self.cfg = cfg
        self.agent = agent

        # build policy net
        self.policy_net = build_policy(cfg, self.agent)

        # build forward dynamics block
        self.dynamics_block = mj_torch_block_factory(agent, 'dynamics').apply
        self.reward_block = mj_torch_block_factory(agent, 'reward').apply

    def forward(self, state):
        """Single pass.
        :param state:
        :return:
        """

        # We're generally using torch.float64 and numpy.float64 for precision, but the net can be trained with
        # torch.float32 -- not sure if this really makes a difference wrt speed or memory, but the default layers
        # seem to be using torch.float32
        # TODO do we stop the flow here?
        action = self.policy_net(state.detach().float()).double()

        # Forward block will drive the simulation forward
        next_state = self.dynamics_block(state, action)

        # The reward is actually calculated in the dynamics_block, here we'll just grab it from the agent
        reward = self.reward_block(state, action)

        return next_state, reward
