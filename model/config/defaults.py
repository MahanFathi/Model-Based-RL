import os
from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Model Configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'Basic'
_C.MODEL.DEVICE = "cpu"
_C.MODEL.WEIGHTS = ""  # should be a path to pth or ckpt file
_C.MODEL.TRAIN_HORIZON = 20

# ---------------------------------------------------------------------------- #
# __Policy Net Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.POLICY = CN()
_C.MODEL.POLICY.HOPPER = CN()
_C.MODEL.POLICY.HOPPER.ARCH = "StochasticPolicy"
_C.MODEL.POLICY.HOPPER.STD = 0.05
_C.MODEL.POLICY.HOPPER.LAYERS = [64, 64, 32]    # a list of hidden layer sizes for output fc. [] means no hidden
_C.MODEL.POLICY.HOPPER.NORM_LAYERS = []         # should be a list of layer indices, example [0, 1, ...]

# ---------------------------------------------------------------------------- #
# Model Configs
# ---------------------------------------------------------------------------- #
_C.MUJOCO = CN()
_C.MUJOCO.ENV = 'HopperEnv'
_C.MUJOCO.GAMMA = 0.98
_C.MUJOCO.HORIZON_STEPS = 100
_C.MUJOCO.POOL_SIZE = CN()
_C.MUJOCO.POOL_SIZE.DYNAMICS = 4
_C.MUJOCO.POOL_SIZE.REWARD = 4

# ---------------------------------------------------------------------------- #
# Experience Replay
# ---------------------------------------------------------------------------- #
_C.EXPERIENCE_REPLAY = CN()
_C.EXPERIENCE_REPLAY.SIZE = 2 ** 14
_C.EXPERIENCE_REPLAY.SHUFFLE = True
_C.EXPERIENCE_REPLAY.ENV_INIT_STATE_NUM = 2 ** 8

# ---------------------------------------------------------------------------- #
# Solver Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 100
_C.SOLVER.BATCH_SIZE = 64

_C.SOLVER.BASE_LR = 0.00001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.CHECKPOINT_PERIOD = 2500
_C.SOLVER.CHECKPOINT_PATH = os.path.join('.', 'checkpoints')

# ---------------------------------------------------------------------------- #
# Output Configs
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.DIR = './output'

# ---------------------------------------------------------------------------- #
# Log Configs
# ---------------------------------------------------------------------------- #
_C.LOG = CN()
_C.LOG.PERIOD = 100
_C.LOG.PLOT = CN()
_C.LOG.PLOT.DISPLAY_PORT = 8097
_C.LOG.PLOT.ITER_PERIOD = 1000  # effective plotting step is _C.LOG.PERIOD * LOG.PLOT.ITER_PERIOD


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
