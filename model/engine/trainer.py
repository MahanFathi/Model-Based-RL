import torch
import numpy as np
from .utils.build import build_state_experience_replay
from utils.logger import setup_logger
from utils.visdom_plots import VisdomLogger


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
    visdom.register_keys(['loss'])
    logger = setup_logger('agent.train', False)
    logger.info("Start training")

    # build and initialize state experience replay
    state_xr = build_state_experience_replay(cfg)
    for _ in range(cfg.SOLVER.BATCH_SIZE):
        env_init_states = [torch.Tensor(agent.reset())] * \
                          int(cfg.EXPERIENCE_REPLAY.ENV_INIT_STATE_NUM / cfg.SOLVER.BATCH_SIZE)
        state_xr.dataset.add_batch(env_init_states)

    iteration = 0
    gamma = cfg.MUJOCO.GAMMA
    for _ in range(cfg.SOLVER.EPOCHS):
        for state_batch in state_xr:
            decay = gamma ** 0
            optimizer.zero_grad()
            reward_sum = torch.zeros((cfg.SOLVER.BATCH_SIZE, 1))
            for _ in range(cfg.MUJOCO.HORIZON_STEPS):
                iteration += 1
                state_batch, reward_batch = model(state_batch)
                reward_sum += decay * reward_batch
                decay *= gamma
                state_xr.dataset.add_batch(state_batch)
            loss = -torch.mean(reward_sum)  # mean over batch
            loss.backward(retain_graph=True)
        print("Reward: \t{}".format(-loss.item()))
        optimizer.step()

        # if iteration % cfg.LOG.PERIOD == 0:
        #     visdom.update({'loss': [loss.item()]})
        #     logger.info("LOSS: \t{}".format(loss))
        #
        # if iteration % cfg.LOG.PLOT.ITER_PERIOD == 0:
        #     visdom.do_plotting()

