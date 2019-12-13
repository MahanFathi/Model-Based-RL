import argparse
import os
from datetime import datetime

from model.engine.trainer import do_training
from model.config import get_cfg_defaults
import utils.logger as lg

def train(cfg, iter):

    # Create output directories
    env_output_dir = os.path.join(cfg.OUTPUT.DIR, cfg.MUJOCO.ENV)
    output_dir = os.path.join(env_output_dir, "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now()))
    output_rec_dir = os.path.join(output_dir, 'recordings')
    output_weights_dir = os.path.join(output_dir, 'weights')
    output_results_dir = os.path.join(output_dir, 'results')
    os.makedirs(output_dir)
    os.mkdir(output_weights_dir)
    os.mkdir(output_results_dir)
    if cfg.LOG.TESTING.ENABLED:
        os.mkdir(output_rec_dir)

    # Create logger
    logger = lg.setup_logger("model.engine.trainer", output_dir, 'logs')
    logger.info("Running with config:\n{}".format(cfg))

    # Repeat for required number of iterations
    for _ in range(iter):
        do_training(
            cfg,
            logger,
            output_results_dir,
            output_rec_dir,
            output_weights_dir
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
        default="train",
        metavar="mode",
        help="'train' or 'test'",
        type=str,
    )
    parser.add_argument(
        "--iter",
        default=1,
        help="Number of iterations",
        type=int
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # TRAIN
    if args.mode == "train":
        train(cfg, args.iter)


if __name__ == "__main__":
    main()
