import torch
from model import archs


def build_model(cfg, agent):
    model_factory = getattr(archs, cfg.MODEL.META_ARCHITECTURE)
    model = model_factory(cfg, agent)
#    if cfg.MODEL.WEIGHTS != "":
#        model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))
    return model
