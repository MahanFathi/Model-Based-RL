import functools
from .functions import soft_lower_bound


def build_soft_lower_bound_fn(policy_cfg):
    soft_lower_bound_fn = functools.partial(soft_lower_bound,
                                            bound=policy_cfg.SOFT_LOWER_STD_BOUND,
                                            threshold=policy_cfg.SOFT_LOWER_STD_THRESHOLD,
                                            )
    return soft_lower_bound_fn
