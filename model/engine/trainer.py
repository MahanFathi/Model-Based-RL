import os
import torch
import numpy as np
from datetime import datetime

from model.engine.tester import do_testing
from utils.logger import setup_logger
from utils.visdom_plots import VisdomLogger
from model.engine.utils import build_state_experience_replay


def do_training(
        cfg,
        model,
        agent,
        optimizer,
        device
):
    # set mode to training for model (aside from policy output, matters for Dropout, BatchNorm, etc.)
    model.train()

    # output directories
    env_output_dir = os.path.join(cfg.OUTPUT.DIR, cfg.MUJOCO.ENV)
    output_dir = os.path.join(env_output_dir, "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now()))
    output_rec_dir = os.path.join(output_dir, 'recordings')
    output_weights_dir = os.path.join(output_dir, 'weights')
    os.makedirs(output_dir)
    os.mkdir(output_weights_dir)

    # get the trainer logger and visdom
    visdom = VisdomLogger(cfg.LOG.PLOT.DISPLAY_PORT)
    visdom.register_keys(['total_loss', 'average_std', 'average_action', "reinforce_loss",
                          "objective_loss", "action", "sd", "action_grad", "sd_grad"])
    logger = setup_logger("model.engine.trainer", output_dir, 'logs')
    logger.info("Start training")
    logger.info("Running with config:\n{}".format(cfg))

    # build and initialize state experience replay
    #state_xr = build_state_experience_replay(cfg)
    #for _ in range(cfg.SOLVER.BATCH_SIZE):
    #    env_init_states = [agent.reset()] * \
    #                      int(cfg.EXPERIENCE_REPLAY.ENV_INIT_STATE_NUM / cfg.SOLVER.BATCH_SIZE)
    #    state_xr.add_batch(env_init_states)

    # wrap screen recorder if testing mode is on
    if cfg.LOG.TESTING.ON:
        visdom.register_keys(['test_reward'])
        # NOTE: wrappers here won't affect the PyTorch MuJoCo blocks
        from gym.wrappers.monitoring.video_recorder import VideoRecorder
        os.mkdir(output_rec_dir)
        video_recorder = VideoRecorder(agent)

    # Start training
    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        batch_loss = torch.empty(cfg.MODEL.POLICY.BATCH_SIZE, cfg.MODEL.POLICY.MAX_HORIZON_STEPS, dtype=torch.float64)
        steps_per_episode = np.zeros((cfg.MODEL.POLICY.BATCH_SIZE,))
        for episode_idx in range(cfg.MODEL.POLICY.BATCH_SIZE):
            # episode_loss = torch.empty(cfg.MODEL.POLICY.MAX_HORIZON_STEPS, dtype=torch.float64)
            # state = state_xr.get_item()

            state = torch.DoubleTensor(agent.reset())
            for step_idx in range(cfg.MODEL.POLICY.MAX_HORIZON_STEPS):
                state, reward = model(state)
                batch_loss[episode_idx, step_idx] = -reward
                #episode_loss[step_idx] = -reward
                #if agent.is_done:
                #    break
                #else:
                #     state_xr.add(state.detach())

            model.policy_net.episode_callback()
            steps_per_episode[episode_idx] = step_idx+1
            #mean = episode_loss[:step_idx + 1].detach().numpy().mean()
            #good_steps = episode_loss.detach().numpy() < mean
            #batch_loss[episode_idx] = torch.sum(episode_loss[:step_idx+1])
            #batch_loss[episode_idx] = torch.sum(episode_loss[good_steps])

        loss = model.policy_net.optimize(batch_loss)

        #batch_loss = episode_loss.sum()
        #loss = torch.mean(batch_loss)
        #loss = torch.sum(episode_loss[:step_idx+1])
       # optimizer.zero_grad()
       # loss.backward()
       # optimizer.step()

        #mean_reward = -loss
        #mean_reward = -np.mean(batch_loss.detach().numpy() / steps_per_episode)

        if epoch_idx % cfg.LOG.PERIOD == 0:
            visdom.update({'episode_length': [np.mean(steps_per_episode)], **loss})
            #if cfg.MODEL.POLICY.VARIATIONAL:
            visdom.update({'average_std': [np.mean(model.policy_net.clamped_sd, axis=(1, 2)).squeeze()]})
            visdom.update({'average_action': [np.mean(model.policy_net.clamped_action, axis=(1, 2)).squeeze()]})
            visdom.set({'action': np.mean(model.policy_net.clamped_action, axis=2).transpose()})
            visdom.set({'sd': np.mean(model.policy_net.clamped_sd, axis=2).transpose()})
            visdom.set({'action_grad': np.cbrt(model.policy_net.mean.grad.detach().numpy().transpose())})
            visdom.set({'sd_grad': np.cbrt(model.policy_net.sd.grad.detach().numpy().transpose())})
            logger.info("REWARD: \t\t{} (iteration {})".format(loss["total_loss"], epoch_idx))

        if epoch_idx % cfg.LOG.PLOT.ITER_PERIOD == 0:
            visdom.do_plotting()

        if epoch_idx % cfg.LOG.CHECKPOINT_PERIOD == 0:
            torch.save(model.state_dict(),
                       os.path.join(output_weights_dir, 'iter_{}.pth'.format(epoch_idx)))

        if cfg.LOG.TESTING.ON:
            if epoch_idx % cfg.LOG.TESTING.ITER_PERIOD == 0:
                #logger.info("TESTING ... ")
                video_recorder.path = os.path.join(output_rec_dir, "iter_{}.mp4".format(epoch_idx))
                test_rewards = []
                #model.policy_net.action_mean.data = action_mean
                #if cfg.MODEL.POLICY.VARIATIONAL:
                #    model.policy_net.action_std.data = action_std
                for _ in range(cfg.LOG.TESTING.COUNT_PER_ITER):
                    test_reward = do_testing(
                        cfg,
                        model,
                        agent,
                        video_recorder,
                        # first_state=state_xr.get_item(),
                    )
                    test_rewards.append(test_reward)
                    model.policy_net.episode_callback()
                mean_reward = np.mean(test_rewards)
                #visdom.update({'test_reward': [np.mean(mean_reward)]})
                #logger.info("REWARD MEAN TEST: \t\t{}".format(mean_reward))
                model.train()
                # video_recorder.close()
