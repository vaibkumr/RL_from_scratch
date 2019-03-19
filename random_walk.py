import numpy as np
import time
import os

class RandomWalk():
    """
    Random walk is a markov reward process or MRP without actions. At each
    state agent will either go right or left with equal probability. No state
    has a non-zero except the 'Right' state. This way intuitively the value of
    the central state shall be the probability of ending up at state 'Right'
    which is 0.5 similarly other values shall be: A = 1/6, B = 2/6, D = 4/6,
    E = 5/6. This is a python implementation of temporal differencing to find
    the optimal values of each state. We aren't concerned with the policy here.
    Refer the book "Reinforcement Learning: An Introductio by Andrew Barto and
    Richard S. Sutton" for more information on this problem.
    - TimeTraveller
    """
    def __init__(self, alpha, episodes, init_value, render_ = False):
        """
        @param alpha learning rate
        @param episodes number of episodes
        @param init_value initial value of value function
        @param render_ render control
        """
        self.render_ = render_
        self.alpha = alpha
        self.episodes = episodes
        self.states = ['Left', 'A', 'B', 'C', 'D', 'E', 'Right']
        self.reward = [0, 0, 0, 0, 0, 0 ,1]
        self.q = np.zeros(7) + init_value

    def render(self, S):
        """ Render the display showing current state """
        a = ['_','_','_','_','_','_','_']
        a[S] = '@'
        print(a)

    def isterminal(self, S):
        """ Returns true if S is terminal """
        if (S <= 0 or S >= 6):
            return True

    def print_q(self):
        """ prints the current values of states """
        for i, q in enumerate(self.q):
            state = self.states[i]
            q = "{:.2f}".format(q)
            print(f"{state} : {q}", end='  ')
        print('\n')

    def find_q(self):
        """ find the optimal Q (value function) """
        for episode in range(self.episodes + 1):
            print(f"Epsode: {episode}")
            S = 3
            while True:
                if self.render_:
                    self.render(S)
                if self.isterminal(S):
                    break
                direction = np.random.choice(['Left', 'Right'])
                if direction == 'Left':
                    S_next = S - 1
                else:
                    S_next = S + 1
                R = self.reward[S_next]
                alpha = self.alpha
                gamma = 1
                self.update_value(S, R, S_next)

                S = S_next
            self.print_q()

    def update_value(self, S, R, S_next):
        """ Update the value function for state S """
        alpha = self.alpha
        gamma = 1
        self.q[S] += alpha*(R + gamma*self.q[S_next] - self.q[S])


def main():
    """ main driver function """
    alpha = 0.1
    episodes = 10000
    walk = RandomWalk(alpha, episodes, 0, True)
    walk.find_q()

if __name__ == "__main__":
    main()
