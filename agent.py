import numpy as np
import numpy as np
def initialize_q_values(strategy, state_space, action_space):
        if strategy == 'zeros':
            return np.zeros((state_space, action_space))
        elif strategy == 'random':
            return np.random.rand(state_space, action_space)
        elif strategy == 'optimistic':
            return np.ones((state_space, action_space)) * 10  # Example optimistic value
        else:
            raise ValueError("Unknown initialization strategy")
        
class Agent(object):
    """The world's simplest agent!"""
        
    def __init__(self, agent_type, state_space, action_space, init_strategy='zeros'):
        self.agent_type = agent_type
        self.state_space = state_space
        self.action_space = action_space
        self.init_strategy = init_strategy
        self.eps = 0.05
        self.alpha = 0.5
        self.gamma = 0.95
        self.prev_state = None
        self.prev_action = None
        
        if self.agent_type == 'Q':
            self.Q = initialize_q_values(self.init_strategy, state_space, action_space)
        elif self.agent_type == 'SARSA':
            self.Q = initialize_q_values(self.init_strategy, state_space, action_space)
            self.prev_state = None
            self.prev_action = None
        elif self.agent_type == 'DoubleQ':
            self.Q1 = initialize_q_values(self.init_strategy, state_space, action_space)
            self.Q2 = initialize_q_values(self.init_strategy, state_space, action_space)
        

    def observe(self, observation, reward, done):
        if self.agent_type == 'Q':
            if self.prev_state is not None and self.prev_action is not None:
                self.update_Q(self.prev_state, self.prev_action, observation, reward)
            if not done:
                action = self.epsilon_greedy_action(observation)
                self.prev_state = observation
                self.prev_action = action
            else:
                self.prev_state = None
                self.prev_action = None
        elif self.agent_type == 'SARSA':
            if self.prev_state is not None and self.prev_action is not None:
                self.update_Q(self.prev_state, self.prev_action, observation, reward)
            if not done:
                action = self.epsilon_greedy_action(observation)
                self.prev_state = observation
                self.prev_action = action
            else:
                self.prev_state = None
                self.prev_action = None
        elif self.agent_type == 'DoubleQ':
            if self.prev_state is not None and self.prev_action is not None:
                self.update_Q(self.prev_state, self.prev_action, observation, reward)
            if not done:
                action = self.epsilon_greedy_action(observation)
                self.prev_state = observation
                self.prev_action = action
            else:
                self.prev_state = None
                self.prev_action = None

    def epsilon_greedy_action(self, state):
        if np.random.rand() < self.eps:
            self.prev_action = np.random.randint(self.action_space)
        else:
            if self.agent_type == 'Q':
                max_action = np.where(self.Q[state] == np.max(self.Q[state]))[0][0]
            elif self.agent_type == 'SARSA':
                if self.prev_state is not None and self.prev_action is not None:
                    max_action = self.prev_action
                else:
                    max_action = np.where(self.Q[state] == np.max(self.Q[state]))[0][0]
            elif self.agent_type == 'DoubleQ':
                for i in range(self.action_space):
                    if self.Q1[state][i] + self.Q2[state][i] > max_action:
                        max_action = i
            self.prev_action = max_action
        return self.prev_action
        
    def update_Q(self, state, action, next_state, reward):
        if self.agent_type == 'Q':
            max_next_action = np.argmax(self.Q[next_state])
            td_target = reward + self.gamma * self.Q[next_state][max_next_action]
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error
            
        elif self.agent_type == 'SARSA':
            next_action = self.epsilon_greedy_action(next_state)
            self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
        elif self.agent_type == 'DoubleQ':
            if np.random.rand() < 0.5:
                max_next_action = np.argmax(self.Q1[next_state])
                self.Q1[state][action] += self.alpha * (reward + self.gamma * self.Q2[next_state][max_next_action] - self.Q1[state][action])
            else:
                max_next_action = np.argmax(self.Q2[next_state])
                self.Q2[state][action] += self.alpha * (reward + self.gamma * self.Q1[next_state][max_next_action] - self.Q2[state][action])

    def act(self, state):
        return self.epsilon_greedy_action(state)

    