import numpy as np
import random

class Agent(object):
    """SARSA agent"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        #self.Q = np.zeros((state_space, action_space), dtype=float)
        self.Q = np.random.random((state_space, action_space))
        #self.Q[state_space - 1][:] = 0
        self.prev_state = 0
        self.prev_action = 0
        self.eps = 0.05
        self.alpha = 0.5
        self.gamma = 0.95
        self.action_picked = False

    def observe(self, observation, reward, done):
        #Add your code here
        max_action = np.where(self.Q[observation] == np.max(self.Q[observation]))[0][0]
        random_action = np.random.choice(np.arange(self.action_space))
        curr_action = np.random.choice([random_action, max_action], p=[self.eps, 1.0 - self.eps])
        self.Q[self.prev_state][self.prev_action] = self.Q[self.prev_state][self.prev_action] + self.alpha * (reward + self.gamma * self.Q[observation][curr_action] - self.Q[self.prev_state][self.prev_action])
        self.action_picked = True
        self.prev_action = curr_action

    def act(self, observation):
        #Add your code here
        self.prev_state = observation

        if self.action_picked:
            return self.prev_action
        else:
            max_action = np.where(self.Q[observation] == np.max(self.Q[observation]))[0][0]
            random_action = np.random.choice(np.arange(self.action_space))
            self.prev_action = np.random.choice([random_action, max_action], p=[self.eps, 1.0 - self.eps])

            return self.prev_action
        #return np.random.randint(self.action_space)