import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
parser.add_argument("--episodes", type=int, help="Number of episodes", default=10000)
parser.add_argument("--runs", type=int, help="Number of runs", default=5)
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)

init_strategies = ['zeros', 'ones', 'random']

try:
    env = gym.make(args.env)
    print("Loaded ", args.env)
    terminal_states = list({
        info[0][1]
        for state, actions in env.env.P.items()
        for a, info in actions.items()
        if info[0][3]
    })
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)
    terminal_states = None



action_dim = env.action_space.n
state_dim = env.observation_space.n


mean_rewards = {}
for strategy in init_strategies:
    all_rewards = []
    for run in range(args.runs):
        rewards = []
        episode = 0
        agent = agentfile.Agent(state_dim, action_dim, terminal_states=terminal_states, init_strategy=strategy)
        observation, info = env.reset()
        while episode < args.episodes:
            print(f'Strategy {strategy}, Run {run}, Episode {episode}, Average: {np.mean(rewards)}', end='\r')
            done = False
            if terminal_states == None:
                action = agent.act(observation)
                observation, reward, done, truncated, info = env.step(action)
                rewards.append(reward)
                agent.observe(observation, reward, done)
                episode += 1
            else:
                while not done:
                    action = agent.act(observation)
                    observation, reward, done, truncated, info = env.step(action)
                    agent.observe(observation, reward, done)
                observation, info = env.reset()
                episode += 1
                rewards.append(reward)
            all_rewards.append(rewards)
        print(f'Strategy {strategy}, Run {run}, Average: {np.mean(rewards)}')
    mean_rewards[strategy] = np.mean(all_rewards, axis=0)
env.close()

window_size = 50
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(env_name, mean_rewards):
    for strategy in init_strategies:
        plt.plot(moving_average(mean_rewards[strategy], window_size), label=strategy)
    plt.title(env_name)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

plot_results(args.env, mean_rewards)
