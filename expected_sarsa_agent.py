import numpy as np

class Agent(object):
    """Expected SARSA agent"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        #self.Q = np.ones((state_space, action_space), dtype=float)
        self.Q = np.random.random((state_space, action_space))
        #self.Q[state_space - 1][:] = 0
        self.prev_state = 0
        self.prev_action = 0
        self.eps = 0.05
        self.alpha = 0.5
        self.gamma = 0.95

    def observe(self, observation, reward, done):
        #Add your code here
        max_action = np.where(self.Q[observation] == np.max(self.Q[observation]))[0][0]
        sum_weights = np.sum(self.Q[observation])

        if sum_weights == 0:
            sum_weights = len(self.Q[observation])

        weights = np.array([x / sum_weights for x in self.Q[observation]])
        values = np.array([weights[i] * self.Q[observation][i] for i in range(self.action_space)])
        expected_value = sum(values)
        self.Q[self.prev_state][self.prev_action] = self.Q[self.prev_state][self.prev_action] + self.alpha * (reward + self.gamma * expected_value - self.Q[self.prev_state][self.prev_action])

    def act(self, observation):
        #Add your code here
        self.prev_state = observation
        max_action = np.where(self.Q[observation] == np.max(self.Q[observation]))[0][0]
        random_action = np.random.choice(np.arange(self.action_space))
        self.prev_action = np.random.choice([random_action, max_action], p=[self.eps, 1.0 - self.eps])

        return self.prev_action
        #return np.random.randint(self.action_space)