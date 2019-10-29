from .base import BasePolicy
from model.layers import FeedForward
import torch
import torch.nn as nn
import torch.distributions as tdist
from model.blocks.utils import build_soft_lower_bound_fn


class VariationalPolicy(BasePolicy):
    def __init__(self, policy_cfg, agent):
        super(VariationalPolicy, self).__init__(policy_cfg, agent)

        self.num_actions = agent.action_space.shape[0]

        self.soft_lower_bound = build_soft_lower_bound_fn(policy_cfg)

        self.std_scaler = policy_cfg.STD_SCALER

        # for deterministic policies the model is just a simple feed forward net
        self.net = FeedForward(
            agent.observation_space.shape[0],
            policy_cfg.LAYERS,
            self.num_actions*2
        )

    def forward(self, s):
        output = self.net(s)
        means = output[:self.num_actions]
        stds = output[self.num_actions:]
        if not self.training:
            return means
        actual_stds = torch.exp(stds) * self.std_scaler
        lower_bounded_stds = self.soft_lower_bound(actual_stds)
        a = tdist.Normal(means, lower_bounded_stds).rsample()  # sample with re-parametrization trick
        return a

    def episode_callback(self):
        pass

    def batch_callback(self):
        pass

