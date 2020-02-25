from solver import build_optimizer
import torch


class DynamicsModel(torch.nn.Module):
    def __init__(self, agent):
        super(DynamicsModel, self).__init__()

        # We're using a simple two layered feedforward net
        self.net = torch.nn.Sequential(
            torch.nn.Linear(agent.observation_space.shape[0] + agent.action_space.shape[0], 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, agent.observation_space.shape[0])
        ).to(torch.device("cpu"))

        self.optimizer = torch.optim.Adam(self.net.parameters(), 0.001)

    def forward(self, state, action):

        # Predict next state given current state and action
        next_state = self.net(torch.cat([state, action]))

        return next_state

