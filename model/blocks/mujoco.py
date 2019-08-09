import torch
from torch import autograd


def mj_torch_block_factory(agent, mode):
    mj_forward = agent.forward_factory(mode)
    mj_gradients = agent.gradient_factory(mode)

    class MjBlock(autograd.Function):

        @staticmethod
        def forward(ctx, state_action):
            ctx.save_for_backward(state_action)
            return torch.Tensor(mj_forward(state_action.detach().numpy()))

        @staticmethod
        def backward(ctx, grad_output):
            state_action, = ctx.saved_tensors
            grad_input = grad_output.clone()
            jacobian = torch.Tensor(mj_gradients(state_action.detach().numpy()))
            return torch.matmul(jacobian, grad_input)

    return MjBlock
