from mujoco import envs
from .utils import MjBlockWrapper, StateWrapper


def build_agent(cfg):
    agent_factory = getattr(envs, cfg.MUJOCO.ENV)
    agent = agent_factory()
    agent = StateWrapper(agent)
    if cfg.MUJOCO.REWARD_SCALE != 1.0:
        from .utils import RewardScaleWrapper
        agent = RewardScaleWrapper(agent, cfg.MUJOCO.REWARD_SCALE)
    if cfg.MUJOCO.CLIP_ACTIONS:
        from .utils import ClipActionsWrapper
        agent = ClipActionsWrapper(agent)
    if cfg.MODEL.POLICY.ARCH == 'TrajOpt':
        from .utils import FixedStateWrapper
        agent = FixedStateWrapper(agent)

    # This should probably be last so we get all wrappers
    agent = MjBlockWrapper(agent)

    return agent
