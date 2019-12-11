from .base import BasePolicy
import torch
from torch.nn.parameter import Parameter
from .strategies import *
import numpy as np


class TrajOpt(BasePolicy):
    """Trajectory Optimization Network"""
    def __init__(self, cfg, agent):
        super(TrajOpt, self).__init__(cfg, agent)

        # Parametrize optimization actions
        action_size = self.agent.action_space.sample().shape[0]
        horizon = cfg.MODEL.POLICY.MAX_HORIZON_STEPS

        #self.action_mean = Parameter(torch.zeros(horizon, action_size))
        #nn.init.normal_(self.action_mean, mean=0.0, std=1.0)
        #self.register_parameter("action_mean", self.action_mean)

        # Get standard deviations as well when doing variational optimization
        #if policy_cfg.VARIATIONAL:
        #    self.action_std = Parameter(torch.empty(horizon, action_size).fill_(-2))
        #    self.register_parameter("action_std", self.action_std)

        self.strategy = PRVO(dim=np.array([action_size, horizon]), nbatch=cfg.MODEL.POLICY.BATCH_SIZE,
                             gamma=cfg.MODEL.POLICY.GAMMA, learning_rate=cfg.SOLVER.BASE_LR)

        # Set index to zero
        self.step_index = 0
        self.episode_index = 0

    def forward(self, s):
        #if self.policy_cfg.VARIATIONAL:
        #    action = torch.distributions.Normal(self.action_mean[self.index], torch.exp(self.action_std[self.index])).rsample()
        #else:
        #    action = self.action_mean[self.index]

        # Sample a new set of actions when required
        if self.step_index == 0:
            self.actions = self.strategy.sample(self.training)

        action = self.actions[:, self.step_index]
        self.step_index += 1
        return action

    def episode_callback(self):
        self.step_index = 0

    def optimize(self, batch_loss):
        return self.strategy.optimize(batch_loss)

