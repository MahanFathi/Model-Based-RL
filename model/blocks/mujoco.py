import torch
from torch import autograd


def mj_torch_block_factory(agent, mode):
    mj_forward = agent.forward_factory(mode)
    mj_gradients = agent.gradient_factory(mode)

    class MjBlock(autograd.Function):

        @staticmethod
        def forward(ctx, state, action):

            # Advance simulation or return reward
            if mode == "dynamics":
                # We need to get a deep copy of simulation data so we can return to this "snapshot"
                # (we can't deepcopy agent.sim.data because some variables are unpicklable)
                # We'll calculate gradients in the backward phase of "reward"
                agent.data.qpos[:] = state[:agent.model.nq].detach().numpy().copy()
                agent.data.qvel[:] = state[agent.model.nq:].detach().numpy().copy()
                agent.data.ctrl[:] = action.detach().numpy().copy()
                agent.data_snapshot = agent.get_snapshot()

                next_state = mj_forward()
                agent.next_state = next_state

                return torch.from_numpy(next_state)

            elif mode == "reward":
                ctx.data_snapshot = agent.data_snapshot
                ctx.reward = agent.reward
                ctx.next_state = agent.next_state
                return torch.Tensor([agent.reward]).double()

            else:
                raise TypeError("mode has to be 'dynamics' or 'gradient'")

        @staticmethod
        def backward(ctx, grad_output):

            # We should need to calculate gradients only once per dynamics/reward cycle
            if mode == "dynamics":
                state_jacobian = 0.95*torch.from_numpy(agent.dynamics_gradients["state"])
                action_jacobian = 0.95*torch.from_numpy(agent.dynamics_gradients["action"])
            elif mode == "reward":
                mj_gradients(ctx.data_snapshot, ctx.next_state, ctx.reward, test=True)
                state_jacobian = torch.from_numpy(agent.reward_gradients["state"])
                action_jacobian = torch.from_numpy(agent.reward_gradients["action"])
            else:
                raise TypeError("mode has to be 'dynamics' or 'reward'")

            return torch.matmul(grad_output, state_jacobian), torch.matmul(grad_output, action_jacobian)

    return MjBlock

