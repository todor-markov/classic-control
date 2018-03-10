import sys
import gym
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from helpers import render_policy, get_policy_mean_reward, get_q_values


class Discrete1LayerPolicy(object):

    def __init__(self, observation_dim, num_actions, num_hidden_units):
        self.num_actions = num_actions
        self.observation_dim = observation_dim
        self._dense1 = tf.layers.Dense(
            num_hidden_units, activation=tf.nn.relu)
        self._dense2 = tf.layers.Dense(num_actions)

    def get_action_logits(self, observations):
        return self._dense2(self._dense1(observations))

    def choose_action(self, observation):
        assert (self.observation_dim == observation.size
                and type(observation == np.ndarray)), (
            'Expected observation to be a numpy.ndarray of size {}'
            .format(self.observation_dim))

        observation_tensor = tf.reshape(observation, [1, self.observation_dim])
        logits = self.get_action_logits(observation_tensor)
        action_probabilities = tf.nn.softmax(logits)[0].numpy()
        return np.argmax(np.random.multinomial(1, action_probabilities))


class PolicyGradientsOptimizer(object):

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def loss_discrete(self, policy, observations, actions, q_values):
        observations = tf.convert_to_tensor(observations)
        logits = policy.get_action_logits(observations)
        negative_likelihoods = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, logits=logits)

        weighted_negative_likelihoods = negative_likelihoods * q_values
        return tf.reduce_mean(weighted_negative_likelihoods)

    def policy_rollout(self, policy, env, n_samples, time_horizon):
        observations = []
        actions = []
        rewards = np.zeros((n_samples, time_horizon))
        n_timesteps = []

        for i in range(n_samples):
            observation = env.reset()
            for t in range(time_horizon):
                action = policy.choose_action(observation)
                observations.append(observation.flatten())
                actions.append(action)

                observation, reward, done, info = env.step(action)
                rewards[i, t] = reward
                if done or t == time_horizon - 1:
                    n_timesteps.append(t+1)
                    break

        q_values = get_q_values(rewards, n_timesteps)
        mean_reward = np.mean(np.sum(rewards, axis=1))

        return (
            np.array(observations),
            np.array(actions),
            np.array(q_values),
            mean_reward)

    def optimize_policy(self,
                        policy,
                        env,
                        n_samples_per_rollout,
                        time_horizon,
                        n_iter=1000,
                        verbose=1):
        grads = tfe.implicit_gradients(self.loss_discrete)
        for i in range(n_iter):
            observations, actions, q_values, mean_reward = self.policy_rollout(
                policy, env, n_samples_per_rollout, time_horizon)

            if verbose >= 1 and (i+1) % 10 == 0:
                print('Iteration {0}. Average reward: {1}'
                      .format(i+1, mean_reward))

            if verbose >= 2 and (i+1) % 10 == 0:
                render_policy(policy, env)

            self.optimizer.apply_gradients(
                grads(policy, observations, actions, q_values))

        return n_iter


if __name__ == '__main__':
    tfe.enable_eager_execution()
    env_name = sys.argv[1]

    env = gym.make(env_name)
    t = time.time()

    policy = Discrete1LayerPolicy(
        observation_dim=np.prod(env.observation_space.shape),
        num_actions=env.action_space.n,
        num_hidden_units=16)

    policy_optimizer = PolicyGradientsOptimizer(
        tf.train.AdamOptimizer(learning_rate=0.1))

    n_iter = policy_optimizer.optimize_policy(
        policy,
        env,
        n_samples_per_rollout=20,
        time_horizon=1000,
        n_iter=50,
        verbose=2)

    print('\nElapsed time: {}'.format(time.time()-t))
    print('Number of iterations: {}'.format(n_iter))
    print('Average reward: {}'
          .format(get_policy_mean_reward(policy, env, n_iter=100)))

    for _ in range(4):
        render_policy(policy, env)

    env.close()
