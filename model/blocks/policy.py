import torch
import torch.nn as nn
from model.layers.feed_forward import FeedForward


class Policy(nn.Module):
    def __init__(self, input_size, middle_sizes, output_size, activation_fn=nn.ReLU):

        super(Policy, self).__init__()
        # for deterministic policies the model is just a simple feed forward net
        self.net = FeedForward(input_size,
                               middle_sizes,
                               output_size,
                               activation_fn=activation_fn)

    def forward(self, x):
        out = self.net(x)
        return out
