from utils.logger import setup_logger
from utils.visdom_plots import VisdomLogger


def do_train(
        cfg,
        model,
        dataloader,
        optimizer,
        device
):
    # set mode to training for model (matters for Dropout, BatchNorm, etc.)
    model.train()

    # get the trainer logger and visdom
    visdom = VisdomLogger(cfg.LOG.PLOT.DISPLAY_PORT)
    visdom.register_keys(['loss'])
    logger = setup_logger('eta_rnn.train', False)
    logger.info("Start training")

    # start the training loop
    for _ in range(cfg.SOLVER.EPOCHS):
        for iteration, (meta_data, sequential_data, sequence_lengths, eta) in enumerate(dataloader):
            meta_data = meta_data.to(device)
            sequential_data = sequential_data.to(device)
            sequence_lengths = sequence_lengths.to(device)
            eta = eta.to(device)

            loss = model(meta_data, sequential_data, sequence_lengths, eta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % cfg.LOG.PERIOD == 0:
                visdom.update({'loss': [loss.item()]})
                logger.info("LOSS: \t{}".format(loss))

            if iteration % cfg.LOG.PLOT.ITER_PERIOD == 0:
                visdom.do_plotting()
