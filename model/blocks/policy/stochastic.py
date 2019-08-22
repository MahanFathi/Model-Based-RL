import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.nn.parameter import Parameter
from model.blocks.utils import build_soft_lower_bound_fn
from model.blocks.policy import DeterministicPolicy


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

        # build soft lower bound function
        self.soft_lower_bound = build_soft_lower_bound_fn(policy_cfg)

    def forward(self, s):
        a_mean = self.mean_net(s)
        if not self.training:
            return a_mean
        a_std_raw = torch.exp(self.logstd)
        a_std = self.soft_lower_bound(a_std_raw)
        a = tdist.Normal(a_mean, a_std).rsample()  # sample with re-parametrization trick
        return a
