from mujoco import envs
from .utils import MjBlockWrapper, TorchTensorWrapper


def build_agent(cfg):
    agent_factory = getattr(envs, cfg.MUJOCO.ENV)
    agent = agent_factory()
    if cfg.MUJOCO.REWARD_SCALE != 1.0:
        from .utils import RewardScaleWrapper
        agent = RewardScaleWrapper(agent, cfg.MUJOCO.REWARD_SCALE)
    agent = MjBlockWrapper(agent)
    if cfg.MUJOCO.CLIP_ACTIONS:
        from .utils import ClipActionsWrapper
        agent = ClipActionsWrapper(agent)
    if cfg.MODEL.POLICY.ARCH is 'TrajOpt':
        from .utils import FixedStateWrapper
        agent = FixedStateWrapper(agent)
    agent = TorchTensorWrapper(agent)
    return agent
