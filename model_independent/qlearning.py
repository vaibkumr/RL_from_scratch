import gym
import numpy as np
import time

"""
Qlearning is an off policy learning python implementation.
This is a python implementation of the qlearning algorithm in the Sutton and
Barto's book on RL. It's called SARSA because - (state, action, reward, state,
action). The only difference between SARSA and Qlearning is that SARSA takes the
next action based on the current policy while qlearning takes the action with
maximum utility of next state.
Using the simplest gym environment for brevity: https://gym.openai.com/envs/FrozenLake-v0/
"""

def init_q(s, a, type="ones"):
    """
    @param s the number of states
    @param a the number of actions
    @param type random, ones or zeros for the initialization
    """
    if type == "ones":
        return np.ones((s, a))
    elif type == "random":
        return np.random.random((s, a))
    elif type == "zeros":
        return np.zeros((s, a))


def epsilon_greedy(Q, epsilon, n_actions, s, train=False):
    """
    @param Q Q values state x action -> value
    @param epsilon for exploration
    @param s number of states
    @param train if true then no random actions selected
    """
    if train or np.random.rand() < epsilon:
        action = np.argmax(Q[s, :])
    else:
        action = np.random.randint(0, n_actions)
    return action

def qlearning(alpha, gamma, epsilon, episodes, max_steps, n_tests):
    """
    @param alpha learning rate
    @param gamma decay factor
    @param epsilon for exploration
    @param max_steps for max step in each episode
    @param n_tests number of test episodes
    """
    env = gym.make('FrozenLake-v0')
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = init_q(n_states, n_actions, type="ones")
    for episode in range(episodes):
        print(f"Episode: {episode}")
        s = env.reset()
        a = epsilon_greedy(Q, epsilon, n_actions, s)
        t = 0
        done = False
        while t < max_steps:
            env.render()
            t += 1
            s_, reward, done, info = env.step(a)
            a_ = np.argmax(Q[s_, :])
            if done:
                Q[s, a] += alpha * ( reward  - Q[s, a] )
            else:
                Q[s, a] += alpha * ( reward + (gamma * Q[s_, a_]) - Q[s, a] )
            s, a = s_, a_
            if done:
                print(f"This episode took {t} timesteps")
                break
    print(f"Here are the Q values:\n{Q}\nTesting now:")
    test_agent(Q, env, n_tests, n_actions)

def test_agent(Q, env, n_tests, n_actions, delay=0.1):
    for test in range(n_tests):
        print(f"Test #{test}")
        s = env.reset()
        done = False
        epsilon = 0
        while True:
            time.sleep(delay)
            env.render()
            a = epsilon_greedy(Q, epsilon, n_actions, s, train=True)
            print(f"Chose action {a} for state {s}")
            s, reward, done, info = env.step(a)
            if done:
                if reward > 0:
                    print("Reached goal!")
                else:
                    print("Shit! dead x_x")
                time.sleep(3)
                break


if __name__ =="__main__":
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.9
    episodes = 10000
    max_steps = 2500
    n_tests = 20
    qlearning(alpha, gamma, epsilon, episodes, max_steps, n_tests)
