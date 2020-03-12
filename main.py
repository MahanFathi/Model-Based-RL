import argparse
import os
from datetime import datetime

import model.engine.trainer
import model.engine.dynamics_model_trainer
from model.config import get_cfg_defaults
import utils.logger as lg
import model.engine.landscape_plot


def train(cfg, iter):

    # Create output directories
    env_output_dir = os.path.join(cfg.OUTPUT.DIR, cfg.MUJOCO.ENV)
    if cfg.OUTPUT.NAME == "timestamp":
        output_dir_name = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())
    else:
        output_dir_name = cfg.OUTPUT.NAME
    output_dir = os.path.join(env_output_dir, output_dir_name)
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
    for i in range(iter):
        agent = model.engine.trainer.do_training(
            cfg,
            logger,
            output_results_dir,
            output_rec_dir,
            output_weights_dir,
            i
        )
        model.engine.landscape_plot.visualise2d(agent, output_results_dir, i)


def train_dynamics_model(cfg, iter):

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
    logger = lg.setup_logger("model.engine.dynamics_model_trainer", output_dir, 'logs')
    logger.info("Running with config:\n{}".format(cfg))

    # Train the dynamics model
    model.engine.dynamics_model_trainer.do_training(
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
        help="'train' or 'test' or 'dynamics'",
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
    elif args.mode == "dynamics":
        train_dynamics_model(cfg, args.iter)


if __name__ == "__main__":
    main()
