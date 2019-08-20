

def do_testing(
        cfg,
        model,
        agent,
        video_recorder,
        first_state=None
):
    state = first_state or agent.reset()
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
