# Model-based Reinforcement Learning

Directly back-propagate into your policy network, from model jacobians calculated in MuJoCo using finite-difference.

### Vanilla Computation Graph
```txt
     +----------+S0+----------+              +----------+S1+----------+
     |                        |              |                        |
     |    +------+   A0   +---v----+         +    +------+   A1   +---v----+
S0+------>+Policy+---+--->+Dynamics+---+---+S1+-->+Policy+---+--->+Dynamics+--->S2  ...
     |    +------+   |    +--------+   |     +    +------+   |    +--------+    |
     |               |                 |     |               |                  |
     |            +--v---+             |     |            +--v---+              |
     +---+S0+---->+Reward+<-----S1-----+     +---+S1+---->+Reward+<-----S2------+
                  +------+                                +------+
```

### Results

<img src="https://imgur.com/iO2vyWa.gif" width="250"> <img src="https://imgur.com/SIPTKLD.gif" width="250"> <img src="https://imgur.com/AfnE9p2.gif" width="250"> 

<img src="https://imgur.com/nOwYQCK.png" width="500"> 
<img src="https://imgur.com/cnTbjIh.png" width="500"> 

### This repo contains:
* Finite-difference calculation of MuJoCo dynamics jacobians in `mujoco-py`
* MuJoCo dynamics as a PyTorch Operation (i.e. forward and backward pass)
* Reward function PyTorch Operation
* Flexible design to wire up your own meta computation graph
* Trajectory Optimization module alongside Policy Networks 
* Flexible design to define your own environment in `gym`
* Fancy logger and monitoring

### Dependencies
Python3.6:
* `torch`
* `mujoco-py`
* `gym`
* `numpy`
* `visdom`

Other:
* Tested w/ `mujoco200`

### Usage
For latest changes:
```bash
git clone -b development git@github.com:MahanFathi/Model-Based-RL.git
```
Run:
```bash
python3 main.py --config-file ./configs/inverted_pendulum.yaml
```
