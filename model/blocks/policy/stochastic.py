import torch
import torch.nn as nn
import torch.distributions as tdist
from .deterministic import DeterministicPolicy


class StochasticPolicy(nn.Module):
    def __init__(self, policy_cfg, agent):
        super(StochasticPolicy, self).__init__()

        self.mean_net = DeterministicPolicy(policy_cfg, agent)
        self.std = policy_cfg.STD

    def forward(self, s):
        a_mean = self.mean_net(s)
        a_std = torch.ones_like(a_mean) * self.std
        a_dist = tdist.Normal(a_mean, a_std)
        a = a_dist.rsample()    # sample with reparametrization trick
        return a
