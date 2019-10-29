import torch
from torch import autograd
from copy import deepcopy

def mj_torch_block_factory(agent, mode):
    mj_forward = agent.forward_factory(mode)
    mj_gradients = agent.gradient_factory(mode)

    class MjBlock(autograd.Function):

        @staticmethod
        def forward(ctx, action):
            #ctx.save_for_backward(action)

            # We need to get a deep copy of simulation data so we can return to this "snapshot"
            # (we can't deepcopy agent.sim.data because some variables are unpiclable)
            data_snapshot = agent.get_snapshot()
            data_snapshot.ctrl = action.detach().numpy()
            ctx.data_snapshot = data_snapshot

            # We might as well save the reward
            reward = mj_forward(action.detach().numpy())
            ctx.reward = reward.copy()
            #print("forward: {}, {}", data_snapshot.time, reward)

            #return torch.Tensor(mj_forward(state_action.detach().numpy()))
            return torch.Tensor(reward)

        @staticmethod
        def backward(ctx, grad_output):
            #state_action, = ctx.saved_tensors
            #action, = ctx.saved_tensors
            #data_snapshot = ctx.data_snapshot
            #data.ctrl[:] = action.detach().numpy()
            grad_input = grad_output.clone()
            #jacobian = torch.Tensor(mj_gradients(state_action.detach().numpy()))
            jacobian = torch.Tensor(mj_gradients(ctx.data_snapshot, ctx.reward))
            #print("backward: {}, {}", ctx.data_snapshot.time, ctx.reward)
            #if mode == "dynamics":
            #    jacobian[:, :4] = 0
            #else:
            #    jacobian[:, :4] = 0
            #print("mode: {}".format(mode))
            #print("grad input: {}".format(grad_input))
            #print("jacobian: {}".format(jacobian))
            #print("output: {}".format(torch.matmul(grad_input, jacobian)))
            #print("")
            return torch.matmul(grad_input, jacobian)

    return MjBlock

