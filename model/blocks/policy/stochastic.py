import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.nn.parameter import Parameter
from .deterministic import DeterministicPolicy


class StochasticPolicy(nn.Module):
    def __init__(self, policy_cfg, agent):
        super(StochasticPolicy, self).__init__()

        # mean network
        self.mean_net = DeterministicPolicy(policy_cfg, agent)

        # variance parameters
        action_size = agent.action_space.sample().shape[0]
        self.std = Parameter(torch.Tensor(action_size, ))
        nn.init.normal_(self.std, mean=policy_cfg.STD_MEAN, std=policy_cfg.STD_STD)
        self.std.data = abs(self.std.data)

    def forward(self, s):
        a_mean = self.mean_net(s)
        a_dist = tdist.Normal(a_mean, self.std)
        a = a_dist.rsample()    # sample with reparametrization trick
        return a
