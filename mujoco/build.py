from mujoco import envs
from .utils import MjBlockWrapper


def build_agent(cfg):
    agent_factory = getattr(envs, cfg.MUJOCO.ENV)
    agent = agent_factory()
    if cfg.MUJOCO.REWARD_SCALE != 1.0:
        from .utils import RewardScaler
        agent = RewardScaler(agent, cfg.MUJOCO.REWARD_SCALE)
    agent = MjBlockWrapper(agent)
    return agent
