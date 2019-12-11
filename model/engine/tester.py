import torch

def do_testing(
        cfg,
        model,
        agent,
        video_recorder,
        samples=None,
        first_state=None
):

    # Let pytorch know we're evaluating a model
    model.eval()

    # We don't need gradients now
    with torch.no_grad():

        if first_state is None:
            state = torch.Tensor(agent.reset(update_episode_idx=False))
        else:
            state = first_state
            agent.set_from_torch_state(state)
        reward_sum = 0.
        episode_iteration = 0
        for step_idx in range(cfg.MODEL.POLICY.MAX_HORIZON_STEPS):
            agent.render()
            # video_recorder.capture_frame()
            #action = model(state)
            #state, reward = model(state, samples[:, step_idx])
            state, reward = model(state)
            #state, reward, done, _ = agent.step(action)
            reward_sum += reward
            #decay *= gamma
            #if agent.is_done:
            #     break
        return reward_sum/episode_iteration
