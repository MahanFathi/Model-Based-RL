import torch


def build_optimizer(cfg, named_parameters):
    params = []
    lr = cfg.SOLVER.BASE_LR
    for key, value in named_parameters:
        if not value.requires_grad:
            continue
        #lr = cfg.SOLVER.BASE_LR / cfg.SOLVER.BATCH_SIZE  # due to funny way gradients are batched
        #weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            #weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if any(x in key for x in ["std", "sd"]):
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.STD_LR_FACTOR
            #weight_decay = 0.
        params += [{"params": [value], "lr": lr}]

    # optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    optimizer = torch.optim.Adam(params, lr, betas=cfg.SOLVER.ADAM_BETAS)
    return optimizer
