import torch
import numpy as np
import os

from model import build_model
import matplotlib.pyplot as pp


def visualise2d(agent, output_dir, iter):

    # Get actions
    actions = agent.policy_net.optimizer.mean.clone().detach()

    # Build the model
    model = build_model(agent.cfg, agent)
    device = torch.device(agent.cfg.MODEL.DEVICE)
    model.to(device)

    # Set mode to test, we're not interested in optimizing the actions now
    model.eval()

    # Choose two random directions
    i_dir = torch.from_numpy(np.random.randn(actions.shape[0])).detach()
    j_dir = torch.from_numpy(np.random.randn(actions.shape[0])).detach()

    # Define range
    i_range = torch.from_numpy(np.linspace(-1, 1, 100)).requires_grad_()
    j_range = torch.from_numpy(np.linspace(-1, 1, 100)).requires_grad_()

    # Collect losses
    loss = torch.zeros((len(i_range), len(j_range)))

    # Collect grads
    #grads = torch.zeros((len(i_range), len(j_range), 2))

    # Need two loops for two directions
    for i_idx, i in enumerate(i_range):
        for j_idx, j in enumerate(j_range):

            # Calculate new parameters
            new_actions = actions + i*i_dir + j*j_dir

            # Set new actions
            model.policy_net.optimizer.mean = new_actions

            # Loop through whole simulation
            state = torch.Tensor(agent.reset())
            for step_idx in range(agent.cfg.MODEL.POLICY.MAX_HORIZON_STEPS):

                # Advance the simulation
                state, reward = model(state.detach())

                # Collect losses
                loss[i_idx, j_idx] += -reward.squeeze()

            # Get gradients
            #loss[i_idx, j_idx].backward(retain_graph=True)
            #grads[i_idx, j_idx, 0] = i_range.grad[i_idx]
            #grads[i_idx, j_idx, 1] = j_range.grad[j_idx]

    # Do some plotting here
    pp.figure(figsize=(12, 12))
    contours = pp.contour(i_range.detach().numpy(), j_range.detach().numpy(), loss.detach().numpy(), colors='black')
    pp.clabel(contours, inline=True, fontsize=8)
    pp.imshow(loss.detach().numpy(), extent=[-1, 1, -1, 1], origin="lower", cmap="RdGy", alpha=0.5)
    pp.colorbar()
    pp.savefig(os.path.join(output_dir, "contour_{}.png".format(iter)))
