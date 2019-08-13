import torch.nn as nn
from model.layers.feed_forward import FeedForward


class DeterministicPolicy(nn.Module):
    def __init__(self, policy_config, agent):
        super(DeterministicPolicy, self).__init__()

        # for deterministic policies the model is just a simple feed forward net
        self.net = FeedForward(
            agent.observation_space.shape[0],
            policy_config.LAYERS,
            agent.action_space.shape[0]
        )

    def forward(self, s):
        a = self.net(s)
        return a
