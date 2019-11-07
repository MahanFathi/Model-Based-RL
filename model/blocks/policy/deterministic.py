from .base import BasePolicy
from model.layers import FeedForward
from torch.nn.parameter import Parameter
import torch


class DeterministicPolicy(BasePolicy):
    def __init__(self, policy_cfg, agent):
        super(DeterministicPolicy, self).__init__(policy_cfg, agent)

        # for deterministic policies the model is just a simple feed forward net
        self.net = FeedForward(
            agent.observation_space.shape[0],
            policy_cfg.LAYERS,
            agent.action_space.shape[0])

        if policy_cfg.VARIATIONAL:
            self.logstd = Parameter(torch.empty(policy_cfg.MAX_HORIZON_STEPS, agent.action_space.shape[0]).fill_(0))
            self.index = 0

    def forward(self, s):
        # Get mean of action value
        a_mean = self.net(s)

        if self.policy_cfg.VARIATIONAL:
            # Get standard deviation of action
            a_std = torch.exp(self.logstd[self.index])
            self.index += 1

            # Sample with re-parameterization trick
            a = torch.distributions.Normal(a_mean, a_std).rsample()

        else:
            a = a_mean

        return a

    def episode_callback(self):
        if self.policy_cfg.VARIATIONAL:
            self.index = 0

    def batch_callback(self):
        pass

