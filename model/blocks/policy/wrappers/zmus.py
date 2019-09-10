import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ZMUSWrapper(nn.Module):
    """Zero-mean Unit-STD States
        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-variance-of-two-groups-given-known-group-variances-mean
    """

    def __init__(self, policy_net, eps=1e-6):
        super(ZMUSWrapper, self).__init__()

        self.eps = eps
        self.policy_net = policy_net
        self.policy_cfg = self.policy_net.policy_cfg

        # Parameters
        state_size = self.policy_net.agent.observation_space.shape
        self.state_mean = Parameter(torch.Tensor(state_size, ))
        self.state_variance = Parameter(torch.Tensor(state_size, ))
        self.state_mean.requires_grad = False
        self.state_variance.requires_grad = False

        # cash
        self.size = 0
        self.ep_states_data = []

        self.first_pass = True

    def _get_state_mean(self):
        return self.state_mean.detach()

    def _get_state_variance(self):
        return self.state_variance.detach()

    def forward(self, s):
        self.size += 1
        self.ep_states_data.append(s)
        if not self.first_pass:
            s = (s - self._get_state_mean()) / \
                (torch.sqrt(self._get_state_variance()) + self.eps)
        return self.policy_net(s)

    def episode_callback(self):
        ep_states_tensor = torch.stack(self.ep_states_data)
        new_data_mean = torch.mean(ep_states_tensor, dim=0)
        new_data_var = torch.var(ep_states_tensor, dim=0)
        if self.first_pass:
            self.state_mean.data = new_data_mean
            self.state_variance.data = new_data_var
            self.first_pass = False
        else:
            n = len(self.ep_states_data)
            mean = self._get_state_mean()
            var = self._get_state_variance()
            new_data_mean_sq = torch.mul(new_data_mean, new_data_mean)
            size = min(self.policy_cfg.FORGET_COUNT_OBS_SCALER, self.size)
            new_mean = ((mean * size) + (new_data_mean * n)) / (size + n)
            new_var = (((size * (var + torch.mul(mean, mean))) + (n * (new_data_var + new_data_mean_sq))) /
                       (size + n) - torch.mul(new_mean, new_mean))
            self.state_mean.data = new_mean
            self.state_variance.data = torch.clamp(new_var, 0.)  # occasionally goes negative, clip
            self.size += n

    def batch_callback(self):
        pass
