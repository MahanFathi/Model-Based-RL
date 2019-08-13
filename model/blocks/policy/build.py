from model.blocks import policy


def build_policy(policy_cfg, agent):
    policy_factory = getattr(policy, policy_cfg.ARCH)
    policy_net = policy_factory(policy_cfg, agent)
    return policy_net

