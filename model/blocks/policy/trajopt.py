from .base import BasePolicy
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class TrajOpt(BasePolicy):
    """Trajectory Optimization Network"""
    def __init__(self, policy_cfg, agent):
        super(TrajOpt, self).__init__(policy_cfg, agent)

        # Parametrize optimization actions
        action_size = self.agent.action_space.sample().shape[0]
        horizon = policy_cfg.MAX_HORIZON_STEPS
        self.action_mean = Parameter(torch.zeros(horizon, action_size))
        #nn.init.normal_(self.action_mean, mean=0.0, std=1.0)
        self.register_parameter("action_mean", self.action_mean)

        # Get standard deviations as well when doing variational optimization
        if policy_cfg.VARIATIONAL:
            self.action_std = Parameter(torch.empty(horizon, action_size).fill_(-2))
            self.register_parameter("action_std", self.action_std)

        # Set index to zero
        self.index = 0

    def forward(self, s):
        if self.policy_cfg.VARIATIONAL:
            action = torch.distributions.Normal(self.action_mean[self.index], torch.exp(self.action_std[self.index])).rsample()
        else:
            action = self.action_mean[self.index]
        self.index += 1
        return action

    def episode_callback(self):
        self.index = 0

    def batch_callback(self):
        pass

