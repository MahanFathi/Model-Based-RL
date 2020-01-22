import torch
import numpy as np
import os

from model.engine.tester import do_testing
import utils.logger as lg
from utils.visdom_plots import VisdomLogger
from mujoco import build_agent
from model import build_model



def do_training(
        cfg,
        logger,
        output_results_dir,
        output_rec_dir,
        output_weights_dir
):
    # Build the agent
    agent = build_agent(cfg)

    # Build the model
    model = build_model(cfg, agent)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # Set mode to training (aside from policy output, matters for Dropout, BatchNorm, etc.)
    model.train()

    # Set up visdom
    if cfg.LOG.PLOT.ENABLED:
        visdom = VisdomLogger(cfg.LOG.PLOT.DISPLAY_PORT)
        visdom.register_keys(['total_loss', 'average_sd', 'average_action', "reinforce_loss",
                              "objective_loss", "sd", "action_grad", "sd_grad", "average_grad"])
        for action_idx in range(model.policy_net.action_dim):
            visdom.register_keys(["action_" + str(action_idx)])

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

            initial_state = torch.DoubleTensor(agent.reset())
            states = []
            states.append(initial_state)
            for step_idx in range(cfg.MODEL.POLICY.MAX_HORIZON_STEPS):
                state, reward = model(states[step_idx])
                batch_loss[episode_idx, step_idx] = -reward
                states.append(state)
                #if agent.is_done:
                #    break

        agent.running_sum = 0
        loss = model.policy_net.optimize(batch_loss)
        output["objective_loss"].append(loss["objective_loss"])
        output["epoch"].append(epoch_idx)

        if epoch_idx % cfg.LOG.PERIOD == 0:

            if cfg.LOG.PLOT.ENABLED:
                visdom.update(loss)
                #visdom.set({'total_loss': loss["total_loss"].transpose()})

                clamped_sd = model.policy_net.get_clamped_sd()
                clamped_action = model.policy_net.get_clamped_action()

                #visdom.update({'average_grad': np.log(torch.mean(model.policy_net.mean._layers["linear_layer_0"].weight.grad.abs()).detach().numpy())})

                if len(clamped_sd) > 0:
                    visdom.update({'average_sd': np.mean(clamped_sd, axis=1)})
                visdom.update({'average_action': np.mean(clamped_action, axis=(1, 2)).squeeze()})

                for action_idx in range(model.policy_net.action_dim):
                    visdom.set({'action_'+str(action_idx): clamped_action[action_idx, :, :]})
                if clamped_sd is not None:
                    visdom.set({'sd': clamped_sd.transpose()})
#                visdom.set({'action_grad': model.policy_net.mean.grad.detach().numpy().transpose()})

            logger.info("REWARD: \t\t{} (iteration {})".format(loss["objective_loss"], epoch_idx))

        if cfg.LOG.PLOT.ENABLED and epoch_idx % cfg.LOG.PLOT.ITER_PERIOD == 0:
            visdom.do_plotting()

        if epoch_idx % cfg.LOG.CHECKPOINT_PERIOD == 0:
            torch.save(model.state_dict(),
                       os.path.join(output_weights_dir, 'iter_{}.pth'.format(epoch_idx)))

        if cfg.LOG.TESTING.ENABLED:
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
