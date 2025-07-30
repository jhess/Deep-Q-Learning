# Navigation Project - Report

Justin Hess

# Introduction

This project uses a Double Deep Q-Learning (DQN) algorithm with Experience Replay to train an agent to navigate and collect yellow bananas and avoid purple bananas.

A reward of 1 is provided for collecting a yellow banana, and a reward of -1 for collecting a purple banana.
The goal the agent is to collect as many yellow bananas as possible while avoiding purple bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
Given this information, the agent has to learn how to best select actions.
Four discrete actions are available:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

# DQN Algorithm

The Double Deep Q-Learning (DQN) algorithm with Experience Replay is used to train an agent. The code is very similar to Lesson 2's Workspace that is on the DQN algorithm with the Lunar agent. In double DQN, two deep Q-learning networks are used: a local network and a target network.
They are the same except that the local network gives the expected Q-value each time step, which updates its value. In the code, its parameters are input into the Adam Optimizer using PyTorch, which updates its weights during training. The target network is used as a reference of the Q-values being evaluated, as it calculates the maximum possible Q-value for the next state.

The target network holds Q-values and weights for time steps that is based on UPDATE_EVERY and then is updated with the weights from the local network, using the soft update function defined. Meanwhile, the local network is being updated (current predicted Q-values and weights) every time step. After the UPDATE_EVERY time steps, the weights are transferred from the local network to the target network, thus periodically updating its values. This procedure compensates.

## Replay Buffer

The Replay Buffer uses a queue for a replay buffer. It randomly selects past experience tuples to avoid correlation among actions and next state. It randomly samples data from the queue for more efficient sample reuse. This avoids temporal correlations during training.

## DQN Update Equation
Q_target = R + gamma * Q_target(next_state, argmax{Q_local(next_state,next_action, w}, w-)

w- is referred to as the 'fixed' weight vector for the target vector that is used as reference and is not updated as frequently as the local network's weights.

The actions are chosen according to the local network, but the target network then predicts the maximum Q-value from that action chosen from the local network. This arg_max is part of the epsilon greedy policy.

The target network is the primary mechanism that prevents Q-values from overshooting during training.
In standard Q-learning, you update Q-values using the Bellman equation:
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
The problem is that both the prediction Q(s,a) and the target r + γ max Q(s',a') use the same network parameters. This creates a "moving target" problem - as you update the network to match the target, the target itself changes, leading to oscillations and instability.
The target network solves this by keeping targets stable; target values r + γ max Q_target(s',a') are computed using fixed parameters θ⁻ that are only updated every UPDATE_EVERY steps. The network being trained (with parameters θ) and the network generating targets (with parameters θ⁻) are decoupled, which breakfs the feedback loop. It also allows gradual convergence, where targets remain consistent long enough for the main network to converge toward them before the targets shift.

## DQN architecture

The DQN network has the states as the input features as it predicts the actions to take as the final outputs. It has 2 hidden dimensions that it projects to (128 and 64, respectively), and it has four layers. The first three layers use non-linear approximation by taking the ReLU of the linear transformations to the hidden dimensions, and the final layer is just a FFN layer that predicts the action values to take.

QNetwork(
  (fc1): Linear(in_features=37, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=4, bias=True)
)

## Agent architecture

Three main functions are used: act, learn, update, soft_update. Epsilon is decayed each time step and is initialized to 1.

The local network agent uses act() to select an action with an epsilon-greedy policy, and it does not update the local network's weights. 

Step() adds the (state, action, next_state, reward) tuple to the replay buffer. Every UPDATE_EVERY number of time steps it randomly samples the BATCH_SIZE amount of tuples from the buffer and calls the learn function. 

Learn() gets the maximum predicted Q values from the target network using the equation above. It gets the expected Q value from the local network and then minimizes the loss using gradient descent and then back propagation update its weights. It then updates the weights of the target network by calling soft_update() which transfers weights from local network to target network.

# Results

Using only the 3 layers with the two diferent hidden dimensions in the DQN achieved a score of 15.0 at about halfway through training, with 1000 episodes being the maximum. The following plot "resultsDQN.png" in the workspace directory demonstrates that the agent is able to receive an average reward of at least 16.


# Future Improvements

It should be investigated to use a Dueling DQN network. It would also be good to investigate prioritized experience replay, instead of just a Replay Buffer that is used here. Finally, using the pixels instead of just the states as inputs would be the final step to perfor, to improve this.