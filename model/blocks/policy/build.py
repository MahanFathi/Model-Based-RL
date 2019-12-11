from model.blocks import policy


def build_policy(cfg, agent):
    policy_factory = getattr(policy, cfg.MODEL.POLICY.ARCH)
    policy_net = policy_factory(cfg, agent)
    #if cfg.MODEL.POLICY.OBS_SCALER:
    #    from .wrappers import ZMUSWrapper
    #    policy_net = ZMUSWrapper(policy_net)
    return policy_net

