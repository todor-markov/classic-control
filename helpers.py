def render_policy(policy, env):
    total_reward = 0
    observation = env.reset()
    while True:
        env.render()
        action = policy.choose_action(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


def get_policy_mean_reward(policy, env, n_iter=10):
    cumulative_reward = 0
    for i_episode in range(n_iter):
        observation = env.reset()
        while True:
            action = policy.choose_action(observation)
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            if done:
                break

    return cumulative_reward / n_iter


def get_q_values(rewards, n_timesteps):
    q_values = rewards - rewards.mean(axis=0)

    for j in range(rewards.shape[1] - 2, -1, -1):
        q_values[:, j] += q_values[:, j+1]

    needed_q_values = []
    for i in range(rewards.shape[0]):
        needed_q_values.extend(q_values[i, 0:n_timesteps[i]])

    return needed_q_values
