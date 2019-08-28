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
_C.MODEL.WEIGHTS = ""  # should be a path to .pth pytorch state dict file

# ---------------------------------------------------------------------------- #
# __Policy Net Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.POLICY = CN()

_C.MODEL.POLICY = CN()
_C.MODEL.POLICY.ARCH = "StochasticPolicy"
_C.MODEL.POLICY.LAYERS = [64, 64, 32, 8]  # a list of hidden layer sizes for output fc. [] means no hidden
_C.MODEL.POLICY.NORM_LAYERS = [0, 1, 2]  # should be a list of layer indices, example [0, 1, ...]
_C.MODEL.POLICY.STD_SCALER = 1e-1
_C.MODEL.POLICY.SOFT_LOWER_STD_BOUND = 1e-4
_C.MODEL.POLICY.SOFT_LOWER_STD_THRESHOLD = 1e-1

# ---------------------------------------------------------------------------- #
# Model Configs
# ---------------------------------------------------------------------------- #
_C.MUJOCO = CN()
_C.MUJOCO.ENV = 'InvertedPendulumEnv'
_C.MUJOCO.ASSETS_PATH = "./mujoco/assets/"
_C.MUJOCO.REWARD_SCALE = 1
_C.MUJOCO.CLIP_ACTIONS = True
_C.MUJOCO.GAMMA = 0.98
_C.MUJOCO.MAX_HORIZON_STEPS = 100
_C.MUJOCO.POOL_SIZE = CN()

# ---------------------------------------------------------------------------- #
# Experience Replay
# ---------------------------------------------------------------------------- #
_C.EXPERIENCE_REPLAY = CN()
_C.EXPERIENCE_REPLAY.SIZE = 2 ** 15
_C.EXPERIENCE_REPLAY.SHUFFLE = True
_C.EXPERIENCE_REPLAY.ENV_INIT_STATE_NUM = 2 ** 15 * 3 / 4

# ---------------------------------------------------------------------------- #
# Solver Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 1000000
_C.SOLVER.BATCH_SIZE = 64

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.STD_LR_FACTOR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

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
_C.LOG.TESTING = CN()
_C.LOG.TESTING.ON = False
_C.LOG.TESTING.ITER_PERIOD = 1000
_C.LOG.TESTING.COUNT_PER_ITER = 2
_C.LOG.CHECKPOINT_PERIOD = 25000


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
