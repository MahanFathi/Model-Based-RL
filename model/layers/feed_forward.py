import torch
import torch.nn as nn
from itertools import tee
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional
import math

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FeedForward(nn.Module):
    def __init__(self, input_size, middle_sizes, output_size, norm_layers=[], activation_fn=nn.LeakyReLU, activation_out=False):
        """
        :param input_size: int input feature size
        :param middle_sizes: [int] list of intermediate hidden state sizes
        :param output_size: int output size of the network
        """
        # TODO: use setattr for linear layers and activations, parameters are empty
        super(FeedForward, self).__init__()
        self._sizes = [input_size] + middle_sizes + [output_size]
        self._layers = OrderedDict()
        for i, (in_size, out_size) in enumerate(pairwise(self._sizes)):
            # Add linear layer
            #linear_layer = Variational(in_size, out_size, bias=True)
            linear_layer = nn.Linear(in_size, out_size, bias=True)
            self.__setattr__('linear_layer_{}'.format(str(i)), linear_layer)
            self._layers.update({'linear_layer_{}'.format(str(i)): linear_layer})
            # Add batch normalization layer
            if i in norm_layers:
                batchnorm_layer = nn.BatchNorm1d(out_size)
                self.__setattr__('batchnorm_layer_{}'.format(str(i)), batchnorm_layer)
                self._layers.update({'batchnorm_layer_{}'.format(str(i)): batchnorm_layer})
            # Add activation layer
            self.__setattr__('activation_layer_{}'.format(str(i)), activation_fn())  # relu for the last layer also makes sense
            self._layers.update({'activation_layer_{}'.format(str(i)): activation_fn()})
        if not activation_out:
            self._layers.popitem()

        self.sequential = nn.Sequential(self._layers)

    def forward(self, x):
        # out = x
        # for i in range(len(self._sizes) - 1):
        #     fc = self.__getattr__('linear_layer_' + str(i))
        #     ac = self.__getattr__('activation_layer_' + str(i))
        #     out = ac(fc(out))
        out = self.sequential(x)
        return out


class Variational(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Variational, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = Parameter(torch.Tensor(out_features, in_features))
        self.weight_std = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias_mean = Parameter(torch.Tensor(out_features))
            self.bias_std = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_std', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        init.constant_(self.weight_std, 0.03)
        if self.bias_mean is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mean)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_mean, -bound, bound)
            init.constant_(self.bias_std, 0.03)

    def forward(self, input):
        neg_weight = self.weight_std.data < 0
        self.weight_std.data[neg_weight] = 0.00001
        weight = torch.distributions.Normal(self.weight_mean, self.weight_std).rsample()
        neg_bias = self.bias_std.data < 0
        self.bias_std.data[neg_bias] = 0.00001
        bias = torch.distributions.Normal(self.bias_mean, self.bias_std).rsample()
        return functional.linear(input, weight, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

