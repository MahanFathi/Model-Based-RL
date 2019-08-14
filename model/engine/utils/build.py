from .experience_replay import StateQueue, make_data_sampler, make_batch_data_sampler, batch_collator, make_data_loader


def build_state_experience_replay_data_loader(cfg):
    state_queue = StateQueue(cfg.EXPERIENCE_REPLAY.SIZE)
    sampler = make_data_sampler(state_queue, cfg.EXPERIENCE_REPLAY.SHUFFLE)
    batch_sampler = make_batch_data_sampler(sampler, cfg.SOLVER.BATCH_SIZE)
    data_loader = make_data_loader(state_queue, batch_sampler, batch_collator)
    return data_loader


def build_state_experience_replay(cfg):
    state_queue = StateQueue(cfg.EXPERIENCE_REPLAY.SIZE)
    return state_queue
