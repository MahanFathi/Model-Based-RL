import torch.nn as nn


class BasePolicy(nn.Module):
    def __init__(self, policy_cfg, agent):
        super(BasePolicy, self).__init__()
        self.policy_cfg = policy_cfg
        self.agent = agent

    def forward(self, s):
        raise NotImplementedError

    def episode_callback(self):
        raise NotImplementedError

    def batch_callback(self):
        raise NotImplementedError
