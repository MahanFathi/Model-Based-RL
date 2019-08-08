from mujoco import envs


def build_agent(cfg):
    agent_factory = getattr(envs, cfg.MUJOCO.ENV)
    agent = agent_factory()
    return agent
