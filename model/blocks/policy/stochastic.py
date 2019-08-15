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
        self.logstd = Parameter(torch.Tensor(action_size, ))
        nn.init.zeros_(self.logstd)
        self.logstd.data = abs(self.logstd.data)

    def forward(self, s):
        a_mean = self.mean_net(s)
        a_std = torch.exp(self.logstd)
        a = tdist.Normal(a_mean, a_std).rsample()  # sample with re-parametrization trick
        return a
