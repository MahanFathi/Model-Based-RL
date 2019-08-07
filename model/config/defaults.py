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
_C.MODEL.META_ARCHITECTURE = 'StraightLSTM'
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = ""  # should be a path to pth or ckpt file

# ---------------------------------------------------------------------------- #
# __RNN Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.RNN = CN()
_C.MODEL.RNN.NUM_LAYERS = 4
_C.MODEL.RNN.HIDDEN_SIZE = 64
_C.MODEL.RNN.DROPOUT = 0.
# ---------------------------------------------------------------------------- #
# __Fully Connected LSTM Output Layer
# ---------------------------------------------------------------------------- #
_C.MODEL.RNN.FC = CN()
# Sequential input network pre-processing
_C.MODEL.RNN.FC.INPUT = CN()            # corresponds to pre-processing sequential data with a feed forward network
_C.MODEL.RNN.FC.INPUT.LAYERS = []    # a list of hidden layer sizes for output fc. [] means no hidden
_C.MODEL.RNN.FC.INPUT.NORM_LAYERS = [0]     # should be a list of layer indices, example [0, 1, ...]
_C.MODEL.RNN.FC.INPUT.OUT_SIZE = 64     # output size of pre-sequential network, which is the input size of LSTM
# Sequential input network pre-processing
_C.MODEL.RNN.FC.OUTPUT = CN()           # corresponds to pre-processing sequential data with a feed forward network
_C.MODEL.RNN.FC.OUTPUT.LAYERS = []   # a list of hidden layer sizes for output fc. [] means no hidden
_C.MODEL.RNN.FC.OUTPUT.NORM_LAYERS = [0]    # should be a list of layer indices, example [0, 1, ...]
_C.MODEL.RNN.FC.OUTPUT.OUT_SIZE = 1     # output should be of size one, which is eta

# ---------------------------------------------------------------------------- #
# __Fully Connected Meta Data Input Layer
# ---------------------------------------------------------------------------- #
_C.MODEL.METADATA = CN()
_C.MODEL.METADATA.FC = CN()
_C.MODEL.METADATA.FC.INITIALIZE_CELL_STATE = False  # this layer initializes hidden state of LSTM. if true, initializes cell state as well.
_C.MODEL.METADATA.FC.LAYERS = [32]  # a list of hidden layer sizes for output fc. [] means no hidden
_C.MODEL.METADATA.FC.NORM_LAYERS = [0]  # should be a list of layer indices, example [0, 1, ...]

# ---------------------------------------------------------------------------- #
# Input Pipeline Configs
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.BATCH_SIZE = 64
_C.INPUT.METADATA = CN()
_C.INPUT.METADATA.FEATURE_SIZE = 5
_C.INPUT.SEQUENTIAL_DATA = CN()
_C.INPUT.SEQUENTIAL_DATA.FEATURE_SIZE = 3

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.FACTORY = 'RouteLogsDataset'    # dataset class
_C.DATASETS.TRAIN_PATH = ''                 # path to dataset
_C.DATASETS.VAL_PATH = ''                   # path to dataset
_C.DATASETS.TEST_PATH = ''                  # path to dataset
_C.DATASETS.SHUFFLE = True                  # load in shuffle fashion

# ---------------------------------------------------------------------------- #
# Dataloader Configs
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 0   # Number of data loading threads

# ---------------------------------------------------------------------------- #
# Solver Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 100

_C.SOLVER.BASE_LR = 0.001
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
