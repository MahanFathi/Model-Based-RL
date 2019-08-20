

def do_testing(
        cfg,
        model,
        agent,
        video_recorder,
):
    state = agent.reset()
    reward_sum = 0.
    while True:
        agent.render()
        video_recorder.capture_frame()
        action = model(state)
        state, reward, done, _ = agent.step(action)
        reward_sum += reward
        if done:
            break
    return reward_sum
