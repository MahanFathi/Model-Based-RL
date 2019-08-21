# Model-based Reinforcement Learning

Directly back-propagate into your policy network, from model jacobians calculated in MuJoCo using finite-difference.

### This repo contains:
* Finite-difference calculation of MuJoCo dynamics jacobians in `mujoco-py`
* MuJoCo dynamics as a PyTorch Operation (i.e. forward and backward pass)
* Reward function PyTorch Operation
* Flexible design to wire up your own meta computation graph
* Flexible design to define your own environment in `gym`

### Vanilla Computation Graph
```txt
     +-----------S0-----------+              +-----------S1-----------+
     |                        |              |                        |
     |    +------+   A0   +---v----+         |    +------+   A1   +---v----+
S0+------>+Policy+---+--->+Dynamics+--------S1--->+Policy+---+--->+Dynamics+--->S2  ...
     |    +------+   |    +--------+         |    +------+   |    +--------+
     |               |                       |               |
     |            +--v---+                   |            +--v---+
     +----S0----->+Reward|                   +----S1----->+Reward|
                  +------+                                +------+
```

### Dependencies
Python3.6:
* `torch`
* `mujoco-py`
* `gym`
* `numpy`

Other:
* Tested w/ `mujoco200`

### Usage
Run:
```bash
python3 main.py --config-file ./configs/hopper.yaml
```
