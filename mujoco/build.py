from mujoco import envs
from .utils import MjBlockWrapper, SnapshotWrapper


def build_agent(cfg):
    agent_factory = getattr(envs, cfg.MUJOCO.ENV)
    agent = agent_factory()
    agent = SnapshotWrapper(agent)
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

    # Maybe we should set opt tolerance to zero so mujoco solvers shouldn't stop early?
    agent.model.opt.tolerance = 0

    return agent
