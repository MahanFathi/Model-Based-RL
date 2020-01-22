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
                return torch.Tensor([agent.reward]).double()

            else:
                raise TypeError("mode has to be 'dynamics' or 'gradient'")

        @staticmethod
        def backward(ctx, grad_output):

            if agent.cfg.MODEL.POLICY.PRIORITISE:
                weight = agent.cfg.MODEL.POLICY.MAX_HORIZON_STEPS - ctx.data_snapshot.step_idx.value
            else:
                weight = 1 / (agent.cfg.MODEL.POLICY.MAX_HORIZON_STEPS - ctx.data_snapshot.step_idx.value)

            # We should need to calculate gradients only once per dynamics/reward cycle
            if mode == "dynamics":

                state_jacobian = torch.from_numpy(agent.dynamics_gradients["state"])
                action_jacobian = torch.from_numpy(agent.dynamics_gradients["action"])

                if agent.cfg.MODEL.POLICY.PRIORITISE:
                    action_jacobian = (1.0 / agent.running_sum) * action_jacobian
                else:
                    action_jacobian = weight * action_jacobian
                    #pass

            elif mode == "reward":

                if agent.cfg.MODEL.POLICY.PRIORITISE:
                    agent.running_sum += weight

                # Calculate gradients, "reward" is always called first
                mj_gradients(ctx.data_snapshot, ctx.next_state, ctx.reward, test=True)
                state_jacobian = torch.from_numpy(agent.reward_gradients["state"])
                action_jacobian = torch.from_numpy(agent.reward_gradients["action"])

                if agent.cfg.MODEL.POLICY.PRIORITISE:
                    state_jacobian = (weight - 1) * state_jacobian
                    action_jacobian = (weight / agent.running_sum) * action_jacobian
                else:
                    #pass
                    action_jacobian = weight * action_jacobian

            else:
                raise TypeError("mode has to be 'dynamics' or 'reward'")

            if True:
                torch.set_printoptions(precision=5, sci_mode=False)
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
                print('weight: {} ---- 1/({} - {})'.format(weight, agent.cfg.MODEL.POLICY.MAX_HORIZON_STEPS, ctx.data_snapshot.step_idx))
                print("")

            ds = torch.matmul(grad_output, state_jacobian)
            da = torch.matmul(grad_output, action_jacobian)

            return ds, da

    return MjBlock

