import torch
from torch import autograd, nn
import numpy as np


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

                ctx.data_snapshot = agent.data_snapshot
#                ctx.reward = agent.reward
#                ctx.next_state = agent.next_state

                return torch.from_numpy(next_state)

            elif mode == "reward":
                ctx.data_snapshot = agent.data_snapshot
                ctx.reward = agent.reward
                ctx.next_state = agent.next_state
                #ctx.step_idx = agent.step_idx
                return torch.Tensor([agent.reward]).double()

            else:
                raise TypeError("mode has to be 'dynamics' or 'gradient'")

        @staticmethod
        def backward(ctx, grad_output):

            # We should need to calculate gradients only once per dynamics/reward cycle
            if mode == "dynamics":
                #tmp = grad_output
                #if agent.previous_grad_output is None:
                #    agent.previous_grad_output = tmp
                #else:
                #    grad_output -= agent.previous_grad_output
                #agent.previous_grad_output = tmp
                state_jacobian = 1.0*torch.from_numpy(agent.dynamics_gradients["state"])
                #th = 1/(200-agent.step_idx)
                action_jacobian = 1.0*torch.from_numpy(agent.dynamics_gradients["action"])

            elif mode == "reward":
                #agent.unwrapped.step_idx = ctx.step_idx
                #print("step_idx in mujoco: {}".format(agent.unwrapped.step_idx))
                mj_gradients(ctx.data_snapshot, ctx.next_state, ctx.reward, test=True)
                state_jacobian = 1.0*torch.from_numpy(agent.reward_gradients["state"])
                action_jacobian = 1.0*torch.from_numpy(agent.reward_gradients["action"])

            else:
                raise TypeError("mode has to be 'dynamics' or 'reward'")

            if False:
                torch.set_printoptions(precision=3, sci_mode=False)
                print("{} {}".format(ctx.data_snapshot.time, mode))
                print("grad_output")
                print(grad_output)
                print("state_jacobian")
                print(state_jacobian)
                print("action_jacobian")
                print(action_jacobian)
                print("grad_output*state_jacobian")
                print(torch.matmul(grad_output, state_jacobian))
                print("grad_output*action_jacobian")
                print(torch.matmul(grad_output, action_jacobian))
                print("")

            #t = 2
            ds = torch.matmul(grad_output, state_jacobian)
            #ds = torch.max(torch.min(ds, torch.DoubleTensor([t])), torch.DoubleTensor([-t]))
            da = torch.matmul(grad_output, action_jacobian)
            #da = torch.max(torch.min(da, torch.DoubleTensor([t])), torch.DoubleTensor([-t]))

            return ds, da

    return MjBlock

