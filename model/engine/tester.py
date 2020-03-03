import torch


def do_testing(
        cfg,
        model,
        agent,
        samples=None,
        first_state=None
):

    # Let pytorch know we're evaluating a model
    model.eval()

    # We don't need gradients now
    with torch.no_grad():

        if first_state is None:
            state = torch.DoubleTensor(agent.reset(update_episode_idx=False))
        else:
            state = first_state
            agent.set_from_torch_state(state)
        reward_sum = 0.
        episode_iteration = 0
        for step_idx in range(cfg.MODEL.POLICY.MAX_HORIZON_STEPS):
            if cfg.LOG.TESTING.RECORD_VIDEO:
                agent.capture_frame()
            else:
                agent.render()
            #state, reward = model(state, samples[:, step_idx])
            state, reward = model(state)
            reward_sum += reward
            if agent.is_done:
                break
        return reward_sum/episode_iteration
