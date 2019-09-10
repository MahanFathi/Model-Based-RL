from .base import BasePolicy
from model.layers import FeedForward


class DeterministicPolicy(BasePolicy):
    def __init__(self, policy_cfg, agent):
        super(DeterministicPolicy, self).__init__(policy_cfg, agent)

        # for deterministic policies the model is just a simple feed forward net
        self.net = FeedForward(
            agent.observation_space.shape[0],
            policy_cfg.LAYERS,
            agent.action_space.shape[0]
        )

    def forward(self, s):
        a = self.net(s)
        return a

    def episode_callback(self):
        pass

    def batch_callback(self):
        pass

