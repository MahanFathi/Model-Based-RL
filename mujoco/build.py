from mujoco import envs
from .utils.wrappers import MjBlockWrapper


def build_agent(cfg):
    agent_factory = getattr(envs, cfg.MUJOCO.ENV)
    agent = agent_factory()
    if cfg.MUJOCO.REWARD_SCALE != 1.0:
        from .utils.wrappers import RewardScaler
        agent = RewardScaler(agent, cfg.MUJOCO.REWARD_SCALE)
    agent = MjBlockWrapper(agent)
    return agent
