from model import meta_architectures


def build_model(cfg):
    model_factory = getattr(meta_architectures, cfg.MODEL.META_ARCHITECTURE)
    model = model_factory(cfg)
    return model
