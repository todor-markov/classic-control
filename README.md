# Classic control

This repository implements vanilla policy gradients. The implementation is based on Sergey Levine's [slides][1] from Berkeley's [Deep Reinforcement Learning][2] course

The policy gradients algorithm is used to train a simple policy to solve the OpenAI classic control environments. The conditional distribution of actions under the policy is parametrized by a neural network with 1 hidden layer of size 16 and ReLU activation. This neural network outputs a real number value for each action (if the action space is discrete) or action dimension (if the action space is continuous)
1. If the action space is discrete, then a softmax layer is applied to the output of the neural network, and the policy draws an action from the resulting distribution
2. If the action space is continuous, then the policy draws an action from a gaussian distribution with mean the output of the neural network, and variable standard deviation, following [Schulman et al. 2015][3] and [Duan et al. 2016][4]

Currently (as of March 18, 2018), this implementation of policy gradients is able to solve the CartPole-v1 and Acrobot-v1 environments fairly quickly (<100 iterations) and the MountainCarContinuous-v0 and Pendulum-v0 more slowly (500 - 1500 iterations). It is unable to solve the MountainCar-v0 environment due to the sparsity of the reward.

Usage: run `python control.py <environment_name>`. While this code is intended for the classic control environments, it can be run with any environment that has an action space of type `Discrete` or `Box` (though it would probably not do very well on environments that are significantly more complex than the classic control ones)

[1]: http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_4_policy_gradient.pdf
[2]: http://rll.berkeley.edu/deeprlcourse/
[3]: https://arxiv.org/abs/1502.05477
[4]: https://arxiv.org/abs/1604.06778