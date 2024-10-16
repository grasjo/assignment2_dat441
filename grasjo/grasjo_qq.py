import numpy as np

class Agent(object):
    """Agent based on double Q-learning"""
    def __init__(self, state_space, action_space, terminal_states=None, init_strategy='zeros'):
        self.action_space = action_space
        observation_space = state_space
        
        if init_strategy == 'zeros':
            self.Q1 = np.zeros((state_space, action_space))
            self.Q2 = np.zeros((state_space, action_space))
        elif init_strategy == 'random':
            self.Q1 = np.random.rand(state_space, action_space)
            self.Q2 = np.random.rand(state_space, action_space)
        elif init_strategy == 'ones':
            self.Q1 = np.ones((state_space, action_space))
            self.Q2 = np.ones((state_space, action_space))
        else:
            raise ValueError("Invalid initialization strategy. Choose from 'zeros', 'random', or 'ones'.")
        
        if terminal_states is not None:
            for s in terminal_states:
                self.Q1[s, :] = 0
                self.Q2[s, :] = 0

        self.epsilon = 0.05
        self.alpha = 0.5
        self.gamma = 0.95

        self.prev_state = None
        self.prev_action = None

        

    def observe(self, observation, reward, done):
        if None not in (self.prev_action, self.prev_state): # only update Q(s_t,a_t) when s_{t+1} and a_{t+1} is available
            if np.random.random() < 0.5:
                self.Q1[self.prev_state, self.prev_action] += self.alpha*(reward + 
                    self.gamma * self.Q2[observation, np.argmax(self.Q1[observation,:])]
                    - self.Q1[self.prev_state, self.prev_action]) 
            else:
                self.Q2[self.prev_state, self.prev_action] += self.alpha*(reward + 
                    self.gamma * self.Q1[observation, np.argmax(self.Q2[observation,:])]
                    - self.Q2[self.prev_state, self.prev_action])
        self.prev_state = observation
        if done: 
            self.prev_state = None
            self.prev_action = None
                    
                                                                            


    def act(self, observation):
        if np.random.random() < self.epsilon: # explore with epsilon-greedy
            self.prev_action = np.random.randint(self.action_space)
        else:
            Q = self.Q1 + self.Q2
            self.prev_action = np.argmax(Q[observation,:])
        return self.prev_action