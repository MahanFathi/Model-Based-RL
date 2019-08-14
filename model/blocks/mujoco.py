import torch
from torch import autograd
import numpy as np
# from torch.multiprocessing import Pool


def mj_torch_block_factory(cfg, agent, mode):
    mj_forward = agent.forward_factory(mode)
    mj_gradients = agent.gradient_factory(mode)

    # pooling fails due to the fact that compiled mujoco models cannot be pickled
    # mj_pool_size = getattr(cfg.MUJOCO.POOL_SIZE, mode.upper())
    # mj_pool = Pool(mj_pool_size)

    class MjBlock(autograd.Function):

        @staticmethod
        def forward(ctx, state_actions):
            ctx.save_for_backward(state_actions)
            states_list = [mj_forward(state_action.detach().numpy()) for state_action in state_actions]
            # states_list = mj_pool.map(mj_forward,
            #                           [state_action.detach().numpy()
            #                            for state_action in state_actions])
            return torch.Tensor(np.stack(states_list))

        @staticmethod
        def backward(ctx, grad_output):
            state_actions, = ctx.saved_tensors
            grad_input = grad_output.clone()
            jacobian_list = [mj_gradients(state_action.detach().numpy()) for state_action in state_actions]
            # jacobian_list = mj_pool.map(mj_gradients,
            #                             [state_action.detach().numpy()
            #                              for state_action in state_actions])
            jacobian = torch.Tensor(np.stack(jacobian_list))
            return torch.matmul(grad_input, jacobian)

    return MjBlock
