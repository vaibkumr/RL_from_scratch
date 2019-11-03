import gym
import numpy as np
import time

"""
TD-lambda python implementation. (There might be a few errors which I am yet to
fix). This is a python implementation of the TD-lambda python algorithm in the
Sutton and Barto's book on RL. Using the simplest gym environment for brevity:
https://gym.openai.com/envs/FrozenLake-v0/
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
    if train or np.random.rand() >= epsilon:
        action = np.argmax(Q[s, :])
    else:
        action = np.random.randint(0, n_actions)
    return action

def td_lambda(alpha, gamma, epsilon, episodes, lmbda, max_steps=2500,
                    n_tests=False, decay=False, render=False):

    env = gym.make('Taxi-v2')
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = init_q(n_states, n_actions, type="ones")
    E_trace = init_q(n_states, n_actions, type="ones") #Same as Q-Table (Basically an accumalation of 'eligibility' of each state)
    total_rewards = []

    for episode in range(episodes):
        print(f"Episode: {episode}")
        # Epsilon decay
        if decay:
            epsilon = epsilon / (episode + 1e-5)

        total_reward = 0
        s = env.reset()
        a = epsilon_greedy(Q, epsilon, n_actions, s)
        done = False
        t = 0
        while t < max_steps:
            if render:
                env.render()
            t += 1
            s_, reward, done, _ = env.step(a)
            a_ = np.argmax(Q[s, :])
            delta = reward + (gamma * Q[s_, a_]) - Q[s, a]
            E_trace[s, a] += 1
            for _state in range(n_states):
                Q[_state, :] += alpha*delta*E_trace[_state, :]
                E_trace[_state, :] = E_trace[_state, :]*gamma*lmbda
            total_reward+= reward
            s, a = s_, a_
            if done:
                if render:
                    print(f"This episode took {t} timesteps and reward {total_reward}")
                total_rewards.append(total_reward)
                break
    if n_tests:
        test_agent(Q, env, n_tests, n_actions)
    return Q, total_rewards


def test_agent(Q, env, n_tests, n_actions, delay=0.1):
    for test in range(n_tests):
        print(f"Test #{test}")
        s = env.reset()
        done = False
        epsilon = 0
        total_reward = 0
        while True:
            time.sleep(delay)
            env.render()
            a = epsilon_greedy(Q, epsilon, n_actions, s, train=True)
            print(f"Chose action {a} for state {s}")
            s, reward, done, info = env.step(a)
            total_reward += reward
            if done:
                print(f"Episode reward: {total_reward}")
                time.sleep(1)
                break


if __name__ =="__main__":
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    episodes = 100
    lmbda = 0.5
    max_steps = 2500
    n_tests = 5
    decay = False
    render = False
    Q, total_rewards = td_lambda(alpha, gamma, epsilon, episodes,
                        lmbda, max_steps, n_tests, decay, render)
    print(total_rewards)
    print(Q)
