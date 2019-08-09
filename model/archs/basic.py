import torch
import torch.nn as nn
from model.blocks.mujoco import mj_torch_block_factory
from model.blocks.policy import Policy


class Basic(nn.Module):
    def __init__(self, cfg, agent):
        """Build the model from the fed config node.

        :param cfg: CfgNode containing the configurations of everything.
        """
        super(Basic, self).__init__()
        self.cfg = cfg
        self.agent = agent

        # get policy config
        env_name = type(self.agent).__name__.split('Env')[0].upper()
        policy_config = getattr(cfg.MODEL.POLICY, env_name)

        # build policy net
        self.policy_net = Policy(
            self.agent.observation_space.shape[0],
            policy_config.LAYERS,
            self.agent.action_space.shape[0],
        )

        # build forward dynamics block
        self.dynamics_block = mj_torch_block_factory(agent, 'dynamics').apply
        self.reward_block = mj_torch_block_factory(agent, 'reward').apply

    def forward(self, state):
        """Single pass.
        :param state:
        :return:
        """

        # get action
        action = self.policy_net(state)
        state_action = torch.cat([state, action])
        next_state = self.dynamics_block(state_action)
        reward = self.reward_block(state_action)
        return next_state, reward


