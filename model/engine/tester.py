import torch

def do_testing(
        cfg,
        model,
        agent,
        video_recorder,
        first_state=None
):

    # Let pytorch know we're evaluating a model
    model.eval()

    # We don't need gradients now
    with torch.no_grad():

        if first_state is None:
            state = torch.Tensor(agent.reset())
        else:
            state = first_state
            agent.set_from_torch_state(state)
        reward_sum = 0.
        episode_iteration = 0
        gamma = cfg.MUJOCO.GAMMA
        decay = gamma ** 0
        while episode_iteration < cfg.MODEL.POLICY.MAX_HORIZON_STEPS:
            episode_iteration += 1
            agent.render()
            # video_recorder.capture_frame()
            #action = model(state)
            state, reward = model(state)
            #state, reward, done, _ = agent.step(action)
            reward_sum += reward * decay
            decay *= gamma
            # if done and first_state is None:
            #     break
        return reward_sum
