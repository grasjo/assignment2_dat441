import numpy as np

class Agent(object):
    """Agent based on expected sarsa"""
    def __init__(self, state_space, action_space, terminal_states=None, init_strategy='zeros'):
        self.action_space = action_space
        self.state_space = state_space
        
        if init_strategy == 'zeros':
            self.Q = np.zeros((state_space, action_space))
        elif init_strategy == 'random':
            self.Q = np.random.rand(state_space, action_space)
        elif init_strategy == 'ones':
            self.Q = np.ones((state_space, action_space))
        else:
            raise ValueError("Invalid initialization strategy. Choose from 'zeros', 'random', or 'ones'.")
        
        if terminal_states is not None:
            for s in terminal_states:
                self.Q[s, :] = 0

        self.epsilon = 0.05
        self.alpha = 0.5
        self.gamma = 0.95

        self.prev_state = None
        self.prev_action = None

        

    def observe(self, observation, reward, done):
        if None not in (self.prev_action, self.prev_state): # only update Q(s_t,a_t) when s_{t+1} and a_{t+1} is available
            expected_q = 0
            max_action = np.argmax(self.Q[self.state, :])
            for action in range(self.action_space):
                if action == max_action:
                    expected_q += (1 - self.epsilon + self.epsilon / self.action_space) * self.Q[self.state, prev_action]
                else:
                    expected_q += (self.epsilon / self.action_space) * self.Q[self.state, prev_action]

            self.Q[self.prev_state, self.prev_action] += self.alpha * (reward 
                + self.gamma * expected_q - self.Q[self.prev_state, self.prev_action])
        self.prev_state = observation  
        if done:
            self.prev_state = None
            self.prev_action = None


    def act(self, observation):
        if np.random.random() < self.epsilon: # explore with epsilon-greedy
            self.prev_action = np.random.randint(self.action_space)
        else:
            self.prev_action = np.argmax(self.Q[observation,:])
        return self.prev_action