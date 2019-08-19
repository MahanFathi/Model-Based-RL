import torch


def soft_lower_bound(x, bound, threshold):
    range = threshold - bound
    return torch.max(x, range * torch.tanh((x - threshold) / range) + threshold)
