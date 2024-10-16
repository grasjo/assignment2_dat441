import argparse
import gymnasium as gym
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()


try:
    env = gym.make(args.env)
    print("Loaded ", args.env)
    terminal_states = list({
        info[0][1]
        for state, actions in env.env.P.items()
        for a, info in actions.items()
        if info[0][3]
    })
    P = env.env.P
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)
    terminal_states = []
    P = {s: {0: [(1, s-1, 0, False)], 
             1: [(0.05, s-1, 0, False), (0.6, s, 0, False), (0.35, s+1, 0, False)]}
        for s in range(env.observation_space.n)
    }
    P[0][0] = [(1, 0, env.small, False)]
    P[env.observation_space.n-1][1] = [(0.4, env.observation_space.n-2, 0, False), (0.6, env.observation_space.n-1, env.large, False)]

action_dim = env.action_space.n
state_dim = env.observation_space.n

Q = np.ones((state_dim, action_dim))
Q[terminal_states, :] = 0
delta = 1
gamma = 0.95
theta = 1e-32
while delta > theta:
    delta = 0
    for s in range(state_dim):
        for a in range(action_dim):
            q = Q[s, a]
            Q[s, a] = np.sum([p * (r + gamma*np.max(Q[s_, :])) for p, s_, r, _ in P[s][a]])
            delta = np.max([delta, np.abs(q - Q[s, a])])

print(Q)
policy = np.array([a for a in np.argmax(Q, axis=1)])
print(policy)
env.close()