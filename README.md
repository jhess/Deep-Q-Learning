# Navigation Project

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

## Getting Started

Download the environment from one of the links below. You need only select the environment that matches your operating system:

Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Dependencies

This is an amended version of the `python/` folder from the [ML-Agents repository](https://github.com/Unity-Technologies/ml-agents).  It has been edited to include a few additional pip packages needed for the Deep Reinforcement Learning Nanodegree program.

Refer to Navigation.ipynb (located in the main workspace directory) and follow the instructions to learn how to train and run the agent.