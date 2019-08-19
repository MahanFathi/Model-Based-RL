import os
import torch
import numpy as np
from datetime import datetime
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
    # set mode to training for model (matters for Dropout, BatchNorm, etc.)
    model.train()

    # get the trainer logger and visdom
    visdom = VisdomLogger(cfg.LOG.PLOT.DISPLAY_PORT)
    visdom.register_keys(['reward'])
    if not os.path.isdir(cfg.OUTPUT.DIR):
        os.mkdir(cfg.OUTPUT.DIR)
    logger = setup_logger("agent.train", cfg.OUTPUT.DIR,
                          '{0:%Y-%m-%d %H:%M:%S}_log'.format(datetime.now()))
    logger.info("Start training")

    logger.info("Running with config:\n{}".format(cfg))

    # build and initialize state experience replay
    state_xr = build_state_experience_replay(cfg)
    for _ in range(cfg.SOLVER.BATCH_SIZE):
        env_init_states = [torch.Tensor(agent.reset())] * \
                          int(cfg.EXPERIENCE_REPLAY.ENV_INIT_STATE_NUM / cfg.SOLVER.BATCH_SIZE)
        state_xr.add_batch(env_init_states)

    gamma = cfg.MUJOCO.GAMMA
    iteration = 0
    for _ in range(cfg.SOLVER.EPOCHS):
        optimizer.zero_grad()
        batch_rewards = []
        for _ in range(cfg.SOLVER.BATCH_SIZE):
            decay = gamma ** 0
            rewards = torch.Tensor()
            # state = torch.Tensor(agent.reset())
            state = state_xr.get_item()
            for _ in range(cfg.MUJOCO.HORIZON_STEPS):
                iteration += 1
                state, reward = model(state)
                state_xr.add(state.detach())
                rewards = torch.cat([rewards, decay * reward])
                decay *= gamma
            loss = -torch.sum(rewards)
            batch_rewards.append(-loss.item())
            loss.backward()
        optimizer.step()
        mean_reward = np.mean(batch_rewards)

        if iteration % cfg.LOG.PERIOD == 0:
            visdom.update({'reward': [mean_reward]})
            logger.info("REWARD: \t{}".format(mean_reward))

        if iteration % cfg.LOG.PLOT.ITER_PERIOD == 0:
            visdom.do_plotting()

