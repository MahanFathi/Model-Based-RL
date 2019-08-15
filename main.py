import os
import argparse
from datetime import datetime
import torch

from model.engine.trainer import do_training
from mujoco import build_agent
from model import build_model
from model.solver import make_optimizer
from model.config import get_cfg_defaults
from utils.logger import setup_logger


def train(cfg):
    # build the agent
    agent = build_agent(cfg)

    # build the agent
    model = build_model(cfg, agent)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # build the optimizer
    optimizer = make_optimizer(cfg, model)

    do_training(
        cfg,
        model,
        agent,
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

    # TRAIN
    train(cfg)


if __name__ == "__main__":
    main()
