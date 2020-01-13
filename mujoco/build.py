from mujoco import envs
from mujoco.utils.wrappers.mj_block import MjBlockWrapper
from mujoco.utils.wrappers.etc import SnapshotWrapper, IndexWrapper, ViewerWrapper


def build_agent(cfg):
    agent_factory = getattr(envs, cfg.MUJOCO.ENV)
    agent = agent_factory(cfg)
#    if cfg.MUJOCO.REWARD_SCALE != 1.0:
#        from .utils import RewardScaleWrapper
#        agent = RewardScaleWrapper(agent, cfg.MUJOCO.REWARD_SCALE)
#    if cfg.MUJOCO.CLIP_ACTIONS:
#        from .utils import ClipActionsWrapper
#        agent = ClipActionsWrapper(agent)
#    if cfg.MODEL.POLICY.ARCH == 'TrajOpt':
#        from .utils import FixedStateWrapper
#        agent = FixedStateWrapper(agent)

    # Make configs accessible through agent
    #agent.cfg = cfg

    # Record video
    agent = ViewerWrapper(agent)

    # Keep track of step, episode, and batch indices
    agent = IndexWrapper(agent, cfg.MODEL.BATCH_SIZE)

    # Grab and set snapshots of data
    agent = SnapshotWrapper(agent)

    # This should probably be last so we get all wrappers
    agent = MjBlockWrapper(agent)

    # Maybe we should set opt tolerance to zero so mujoco solvers shouldn't stop early?
    agent.model.opt.tolerance = 0

    return agent
