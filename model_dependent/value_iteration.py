"""
This is the python implementation of value epsation algorithm
(Reinforcement learning) from the infamous Sutton and Barto's book on RL.
The algorithm is applied on the famous 3x4 grid robot model. Since value interation
is a model dependent algorithm, I am only testing it on a very simple and small
model whose transition table is stored in T.npy (load using numpy)
"""

import numpy as np
import copy
import time

def getUtil(reward, V, gamma, T, n_actions, state):
    reward = reward[state]
    u = np.zeros(n_actions)
    for action in range(n_actions):
        u[action] = np.multiply(T[state, :, action], V).sum()
    return reward + gamma * u.max()

def getPolicy(reward, V, gamma, T, n_actions, n_states):
    policy = np.zeros(n_states)
    for state in range(n_states):
        policy[state] = np.argmax(gamma * np.dot(T[state, :, :].T, V).T)
    return policy


def valueIteration(gamma, theta, r, silent = False):
    start = time.time()
    living_reward, goal_reward, death_reward = r
    rewards = np.ones(12) * living_reward
    rewards[3] = goal_reward
    rewards[7] = death_reward
    rewards[5] = 0 #dead state, no reward here.
    rewards = np.array([-0.04, -0.04, -0.04,  +1.0,
                        -0.04,   0.0, -0.04,  -1.0,
                        -0.04, -0.04, -0.04, -0.04])
    T = np.load('T.npy')
    n_states = 3 * 4
    n_actions = 4
    v1 = v0 = np.zeros(n_states)
    eps = 0
    while(1): #Do until convergence
        eps += 1
        v0 = v1.copy()
        delta = 0
        for state in range(n_states):
            v1[state] = getUtil(rewards, v1, gamma, T, n_actions, state)
            delta = max(delta, np.abs(v0[state] - v1[state]))
        #Convergence check
        if delta < theta:
            policy = getPolicy(rewards, v1, gamma, T, n_actions, n_states)
            end = time.time()
            if not silent:
                print(f"Number of episodes required for convergence: {eps}")
                print(f"Values are:\n {v1.reshape(3, 4)}")
                print(f"Policy is:\n {policy.reshape(3, 4)}")
                print(f"Time taken: {end-start}")
            return(eps, end-start)
            break

if __name__ == "__main__":
    """
    This is what the grid looks like:
    A A A G
    A N A D
    A A A A
    G is goal (positive reward)
    D is death (negative reward)
    N is neutral (no reward)
    A is any normal state (a negative living reward for each of A)
    """
    living_reward, goal_reward, death_reward = 0.05, 1, -1
    r = [living_reward, goal_reward, death_reward]
    gamma = 1
    theta = 1e-10
    valueIteration(gamma, theta, r)
