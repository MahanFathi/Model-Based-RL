import torch
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

    gamma = cfg.MUJOCO.GAMMA
    iteration = 0
    for _ in range(cfg.SOLVER.EPOCHS):
        decay = gamma ** 0
        rewards = torch.Tensor()
        state = torch.Tensor(agent.reset())
        # while not agent.is_done(state.detach().numpy()):
        for _ in range(cfg.MUJOCO.HORIZON_STEPS):
            iteration += 1
            state, reward = model(state)
            rewards = torch.cat([rewards, decay * reward])
            decay *= gamma
        loss = -torch.sum(rewards)
        print("Reward: \t{}".format(-loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if iteration % cfg.LOG.PERIOD == 0:
        #     visdom.update({'loss': [loss.item()]})
        #     logger.info("LOSS: \t{}".format(loss))
        #
        # if iteration % cfg.LOG.PLOT.ITER_PERIOD == 0:
        #     visdom.do_plotting()

