from .base import BasePolicy
from model.layers import FeedForward
import torch
import torch.nn as nn
import torch.distributions as tdist
from model.blocks.utils import build_soft_lower_bound_fn


class StochasticPolicy(BasePolicy):
    def __init__(self, policy_cfg, agent):
        super(StochasticPolicy, self).__init__(policy_cfg, agent)

        # Get number of actions
        self.num_actions = agent.action_space.shape[0]

        # The network outputs a gaussian distribution
        self.net = FeedForward(
            agent.observation_space.shape[0],
            policy_cfg.LAYERS,
            self.num_actions*2
        )

    def forward(self, s):
        # Get means and logs of standard deviations
        output = self.net(s)
        means = output[:self.num_actions]
        log_stds = output[self.num_actions:]

        # Return only means when testing
        if not self.training:
            return means

        # Get the actual standard deviations
        stds = torch.exp(log_stds)

        # Sample with re-parameterization trick
        a = tdist.Normal(means, stds).rsample()

        return a

    def episode_callback(self):
        pass

    def batch_callback(self):
        pass

