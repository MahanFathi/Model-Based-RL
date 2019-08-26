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

    # get the trainer logger and visdom
    visdom = VisdomLogger(cfg.LOG.PLOT.DISPLAY_PORT)
    visdom.register_keys(['train_reward'])
    if not os.path.isdir(cfg.OUTPUT.DIR):
        os.mkdir(cfg.OUTPUT.DIR)
    logger = setup_logger("agent.train", cfg.OUTPUT.DIR,
                          '{0:%Y-%m-%d %H:%M:%S}_log'.format(datetime.now()))
    logger.info("Start training")
    logger.info("Running with config:\n{}".format(cfg))

    # build and initialize state experience replay
    state_xr = build_state_experience_replay(cfg)
    for _ in range(cfg.SOLVER.BATCH_SIZE):
        env_init_states = [agent.reset()] * \
                          int(cfg.EXPERIENCE_REPLAY.ENV_INIT_STATE_NUM / cfg.SOLVER.BATCH_SIZE)
        state_xr.add_batch(env_init_states)

    # wrap screen recorder if testing mode is on
    if cfg.LOG.TESTING.ON:
        visdom.register_keys(['test_reward'])
        # NOTE: wrappers here won't affect the PyTorch MuJoCo blocks
        from gym.wrappers.monitoring.video_recorder import VideoRecorder
        output_rec_dir = os.path.join(cfg.OUTPUT.DIR, '{0:%Y-%m-%d %H:%M:%S}_rec'.format(datetime.now()))
        os.mkdir(output_rec_dir)
        video_recorder = VideoRecorder(agent)

    iteration = 0
    gamma = cfg.MUJOCO.GAMMA
    for _ in range(cfg.SOLVER.EPOCHS):
        optimizer.zero_grad()
        batch_rewards = []
        for _ in range(cfg.SOLVER.BATCH_SIZE):
            decay = gamma ** 0
            episode_reward = 0.
            # state = state_xr.get_item()
            state = agent.reset()
            for _ in range(cfg.MUJOCO.MAX_HORIZON_STEPS):
                iteration += 1
                state, reward = model(state)
                episode_reward += decay * reward
                decay *= gamma
                # if agent.is_done(state):
                #     break
                # else:
                #     state_xr.add(state.detach())
            loss = -episode_reward
            batch_rewards.append(-loss.item())
            loss.backward()
        optimizer.step()
        mean_reward = np.mean(batch_rewards)

        if iteration % cfg.LOG.PERIOD == 0:
            visdom.update({'train_reward': [mean_reward]})
            logger.info("REWARD: \t\t{}".format(mean_reward))

        if iteration % cfg.LOG.PLOT.ITER_PERIOD == 0:
            visdom.do_plotting()

        if cfg.LOG.TESTING.ON:
            if iteration % cfg.LOG.TESTING.ITER_PERIOD == 0:
                logger.info("TESTING ... ")
                model.eval()
                video_recorder.path = os.path.join(output_rec_dir, "iter_{}.mp4".format(iteration))
                test_rewards = []
                for _ in range(cfg.LOG.TESTING.COUNT_PER_ITER):
                    test_reward = do_testing(
                        cfg,
                        model,
                        agent,
                        video_recorder,
                        # first_state=state_xr.get_item(),
                    )
                    test_rewards.append(test_reward)
                mean_reward = np.mean(test_rewards)
                visdom.update({'test_reward': [np.mean(mean_reward)]})
                logger.info("REWARD MEAN TEST: \t\t{}".format(mean_reward))
                model.train()
                # video_recorder.close()


