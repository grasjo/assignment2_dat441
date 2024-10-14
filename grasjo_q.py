import numpy as np

class Agent(object):
    """Agent based on sarsa learning"""
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

        self.state = None
        self.prev_state = None
        self.action = None
        self.prev_action = None

        

    def observe(self, observation, reward, done):
        if None not in (self.prev_action, self.prev_state): # only update Q(s_t,a_t) when s_{t+1} and a_{t+1} is available
            if done: # handle edge-case where s_{t+1} is terminal
                self.Q[self.prev_state, self.prev_action] += self.alpha*(reward - self.Q[self.prev_state, self.prev_action])
                self.state = None
                self.prev_state = None
                self.action = None
                self.prev_action = None
            else:
                self.Q[self.prev_state, self.prev_action] += self.alpha*(reward + self.gamma * np.max(self.Q[self.state, :]) 
                                                                         - self.Q[self.prev_state, self.prev_action])


    def act(self, observation):
        self.prev_state = self.state
        self.prev_action = self.action
        self.state = observation
        if np.random.random() < self.epsilon: # explore with epsilon-greedy
            self.action = np.random.randint(self.action_space)
        else:
            self.action = np.argmax(self.Q[self.state,:])
        return self.action