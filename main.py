import os
import argparse
from datetime import datetime
import torch
from model import build_model
from model.solver import make_optimizer
from model.config import get_cfg_defaults
from model.engine import do_train, do_inference
from data import make_data_loader
from utils.logger import setup_logger


def train(cfg):
    # build the model
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # build the optimizer
    optimizer = make_optimizer(cfg, model)

    # build the dataloader
    dataloader = make_data_loader(cfg, 'train')

    # start the training procedure
    do_train(
        cfg,
        model,
        dataloader,
        optimizer,
        device
    )


def inference(cfg):
    pass


def main():
    parser = argparse.ArgumentParser(description="PyTorch model-based RL.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="file",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--mode",
        default="test",
        metavar="mode",
        help="'train' or 'test'",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # setup the logger
    if not os.path.isdir(cfg.OUTPUT.DIR):
        os.mkdir(cfg.OUTPUT.DIR)
    logger = setup_logger("agent.train", cfg.OUTPUT.DIR,
                          '{0:%Y-%m-%d %H:%M:%S}_log'.format(datetime.now()))
    logger.info(args)
    logger.info("Running with config:\n{}".format(cfg))

    # TRAIN
    train(cfg)


if __name__ == "__main__":
    main()
