# script that takes a string of a policy and draws the actions on a grid

import numpy as np
import matplotlib.pyplot as plt

# plot a grid with arrows corresponding to the actions in the policy
def draw_policy(policy, shape, terminal_states=None, goal=None):
    policy = np.array(policy).reshape(shape)
    action_dict = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    plt.figure()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i, j) in terminal_states:
                plt.plot(j, i, 'bo', markersize=15)  # Blue circle for terminal states
            elif (i, j) == goal:
                plt.plot(j, i, 'gs', markersize=15)  # Green square for goal state
            else:
                plt.text(j, i, action_dict[policy[i, j]], ha='center', va='center')
    plt.xlim(-0.5, shape[1] - 0.5)
    plt.ylim(-0.5, shape[0] - 0.5)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()

draw_policy([0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0], (4,4), [(1,1),(1,3),(2,3),(3,0)],(3,3))