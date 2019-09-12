from .base import BasePolicy
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class TrajOpt(BasePolicy):
    """Trajectory Optimization Network"""
    def __init__(self, policy_cfg, agent):
        super(TrajOpt, self).__init__(policy_cfg, agent)

        # parametrize optimization actions
        action_size = self.agent.action_space.sample().shape[0]
        horizon = policy_cfg.MAX_HORIZON_STEPS
        self.optim_actions = [Parameter(torch.Tensor(action_size, )) for _ in range(horizon)]
        for i, optim_action in enumerate(self.optim_actions):
            self.register_parameter("optim_action_{}".format(i), optim_action)
            nn.init.normal_(optim_action)

        # set index to zero for the first run
        self.index = 0

    def forward(self, s):
        action = self.optim_actions[self.index]
        self.index += 1
        return action

    def episode_callback(self):
        self.index = 0

    def batch_callback(self):
        pass

