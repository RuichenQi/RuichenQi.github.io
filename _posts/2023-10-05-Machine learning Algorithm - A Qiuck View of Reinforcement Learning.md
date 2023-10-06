---
layout: post
title: Machine learning Algorithm - A Qiuck View of Reinforcement Learning
subtitle: This article introduces the concept and limitations of reinforcement learning.
categories: AI
tags: Machine_Learning_Algorithms
---

## Concept of reinforcement learning

Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. It is one of the three basic machine learning paradigms, alongside supervised learning and unsupervised learning.

## Reinforcement Learning Formalism

The goal of reinforcement learning is to learn an optimal policy that guides an agent's actions in an environment to maximize the cumulative reward it receives over time. After taking an action, the agent receives feedback (positive or negative) from the environment. It uses this feedback to update its policy and improve decision-making.

Here are two examples of the reward system:
- Autonomous helicopter: 
  - Positive reward: helicopter flying well +1.
  - Negative reward: helicopter flying poorly -1000.
- Mars rover
  - Two end states: to make it goone way, make the reward higher.
  - Make the intermediate reward 0 so that it will not stop there.
  - Gradually reduce the reward of each step ( by using discount factor γ ) so that it can choose the best solution.
  - Once it gets the terminal, nothing futher happens ( No further rewards ).
  - Data structure we use: ( state, action, reward, next_state ).

## Markov decision process (MDP)
The environment in RL is typically modeled as a Markov decision process, which consists of states, actions, transition probabilities, and rewards. The agent's goal is to find a policy that maximizes expected cumulative rewards. The markov decision process is shown as follows:

![markov](https://ruichenqi.github.io/assets/images/AI/2/markov.png)

## Bellman equation

Let's make some definitions:
- s: current state
- a: current action
- s': state you get to after taking action a
- a': action that you will take in state s'

Then we can use bellman equation as follows:

![bellman](https://ruichenqi.github.io/assets/images/AI/2/bellman.png)

In this equation, R(s) is the reward you get right away, γ is discount factor, and maxQ(s', a') is the best possible return from state s'.

## Limitations of reinforcement learning

Limitations of reinforcement learning are as follows:
- Much easier to get to work in a simulation rather than in a real robot.
- Far fewer applications than supervised and unsupervised learning.
- It needs a lot of data and a lot of computation, which can be costly and time-consuming.
- It is highly dependent on the quality of the reward function, which can be hard to design and optimize.
- It needs extensive experience to interact with the environment, which can be limited by the dynamics and latency of the environment.

