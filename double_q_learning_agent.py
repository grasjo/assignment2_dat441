import numpy as np
import random

class Agent(object):
    """Q-learning agent"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        #self.Q1 = np.ones((state_space, action_space), dtype=float)
        self.Q1 = np.random.random((state_space, action_space))
        #self.Q2 = np.ones((state_space, action_space), dtype=float)
        self.Q2 = np.random.random((state_space, action_space))
        #self.Q1[state_space - 1][:] = 0
        #self.Q2[state_space - 1][:] = 0
        self.prev_state = 0
        self.prev_action = 0
        self.eps = 0.05
        self.alpha = 0.5
        self.gamma = 0.95

    def observe(self, observation, reward, done):
        #Add your code here
        if random.uniform(0, 1) < 0.5:
            max_action = np.where(self.Q1[observation] == np.max(self.Q1[observation]))[0][0]
            self.Q1[self.prev_state][self.prev_action] = self.Q1[self.prev_state][self.prev_action] + self.alpha * (reward + self.gamma * self.Q2[observation][max_action] - self.Q1[self.prev_state][self.prev_action])
        else:
            max_action = np.where(self.Q2[observation] == np.max(self.Q2[observation]))[0][0]
            self.Q2[self.prev_state][self.prev_action] = self.Q2[self.prev_state][self.prev_action] + self.alpha * (reward + self.gamma * self.Q1[observation][max_action] - self.Q2[self.prev_state][self.prev_action])

    def act(self, observation):
        #Add your code here
        self.prev_state = observation
        max_action = 0

        for i in range(self.action_space):
            if self.Q1[observation][i] + self.Q2[observation][i] > max_action:
                max_action = i

        random_action = np.random.choice(np.arange(self.action_space))
        self.prev_action = np.random.choice([random_action, max_action], p=[self.eps, 1.0 - self.eps])

        return self.prev_action
        #return np.random.randint(self.action_space)