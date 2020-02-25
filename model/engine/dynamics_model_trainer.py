import torch
import numpy as np
import os

from model.engine.tester import do_testing
import utils.logger as lg
from utils.visdom_plots import VisdomLogger
from mujoco import build_agent
from model import build_model
from model.blocks.policy.dynamics import DynamicsModel
import torchviz


def do_training(
        cfg,
        logger,
        output_results_dir,
        output_rec_dir,
        output_weights_dir
):
    # Build the agent
    agent = build_agent(cfg)

    # Build a forward dynamics model
    dynamics_model = DynamicsModel(agent)

    # Set mode to training (aside from policy output, matters for Dropout, BatchNorm, etc.)
    dynamics_model.train()

    # Set up visdom
    if cfg.LOG.PLOT.ENABLED:
        visdom = VisdomLogger(cfg.LOG.PLOT.DISPLAY_PORT)
        visdom.register_keys(['total_loss', 'average_sd', 'average_action', "reinforce_loss",
                              "objective_loss", "sd", "action_grad", "sd_grad", "actions"])

    # wrap screen recorder if testing mode is on
    if cfg.LOG.TESTING.ENABLED:
        if cfg.LOG.PLOT.ENABLED:
            visdom.register_keys(['test_reward'])

    # Collect losses here
    output = {"epoch": [], "objective_loss": []}

    # Start training
    for epoch_idx in range(cfg.MODEL.EPOCHS):
        batch_loss = torch.empty(cfg.MODEL.BATCH_SIZE, cfg.MODEL.POLICY.MAX_HORIZON_STEPS, dtype=torch.float64)
        batch_loss.fill_(np.nan)

        for episode_idx in range(cfg.MODEL.BATCH_SIZE):

            # Generate "random walk" set of actions (separately for each dimension)
            action = torch.zeros(agent.action_space.shape, dtype=torch.float64)
            #actions = np.zeros((agent.action_space.shape[0], cfg.MODEL.POLICY.MAX_HORIZON_STEPS))
            actions = []

            initial_state = torch.Tensor(agent.reset())
            predicted_states = []
            real_states = []
            corrections = []
            for step_idx in range(cfg.MODEL.POLICY.MAX_HORIZON_STEPS):

                # Generate random actions
                action = action + 0.1*(2*torch.rand(agent.action_space.shape) - 1)

                # Clamp to [-1, 1]
                action.clamp_(-1, 1)

                # Save action
                actions.append(action)

                previous_state = torch.from_numpy(agent.unwrapped._get_obs())

                # Advance the actual simulation
                next_state, _, _, _ = agent.step(action)
                next_state = torch.from_numpy(next_state)
                real_states.append(next_state)

                # Advance with learned dynamics simulation
                pred_next_state = dynamics_model(previous_state.float(), action.float()).double()

                batch_loss[episode_idx, step_idx] = torch.pow(next_state - pred_next_state, 2).mean()
                #if agent.is_done:
                #    break

        #dot = torchviz.make_dot(pred_next_state, params=dict(dynamics_model.named_parameters()))

        loss = torch.sum(batch_loss)
        dynamics_model.optimizer.zero_grad()
        loss.backward()
        dynamics_model.optimizer.step()

        output["objective_loss"].append(loss.detach().numpy())
        output["epoch"].append(epoch_idx)

        if epoch_idx % cfg.LOG.PERIOD == 0:

            if cfg.LOG.PLOT.ENABLED:
                visdom.update({"total_loss": loss.detach().numpy()})
                visdom.set({'actions': torch.stack(actions).detach().numpy()})
                #visdom.set({'total_loss': loss["total_loss"].transpose()})
                #visdom.update({'average_grad': np.log(torch.mean(model.policy_net.mean._layers["linear_layer_0"].weight.grad.abs()).detach().numpy())})

            logger.info("REWARD: \t\t{} (iteration {})".format(loss.detach().numpy(), epoch_idx))

        if cfg.LOG.PLOT.ENABLED and epoch_idx % cfg.LOG.PLOT.ITER_PERIOD == 0:
            visdom.do_plotting()

#        if epoch_idx % cfg.LOG.CHECKPOINT_PERIOD == 0:
#            torch.save(model.state_dict(),
#                       os.path.join(output_weights_dir, 'iter_{}.pth'.format(epoch_idx)))

        if False:#cfg.LOG.TESTING.ENABLED:
            if epoch_idx % cfg.LOG.TESTING.ITER_PERIOD == 0:

                # Record if required
                agent.start_recording(os.path.join(output_rec_dir, "iter_{}.mp4".format(epoch_idx)))

                test_rewards = []
                for _ in range(cfg.LOG.TESTING.COUNT_PER_ITER):
                    test_reward = do_testing(
                        cfg,
                        model,
                        agent,
                        # first_state=state_xr.get_item(),
                    )
                    test_rewards.append(test_reward)

                # Set training mode on again
                model.train()

                # Close the recorder
                agent.stop_recording()

    # Save outputs into log folder
    lg.save_dict_into_csv(output_results_dir, "output", output)

    # Save model
    torch.save(dynamics_model.state_dict(), os.path.join(output_weights_dir, "final_weights.pt"))
