import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from riverswim import RiverSwim

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--agenttype", type=str, help="Agent type", default="Q")
parser.add_argument("--strategy", type=str, help="Strategy", default="random")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)



try:
    env = gym.make(args.env)
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)


rewards = []
action_dim = env.action_space.n
state_dim = env.observation_space.n
print(f"State dim:{state_dim}, Action dim:{action_dim}")
print(args.agenttype, state_dim, action_dim, args.strategy)
agent = agentfile.Agent(args.agenttype, state_dim, action_dim, args.strategy)

# observation = env.reset()
# num_runs = 1000
# for _ in range(num_runs): 
#     #env.render()
#     # Handle different observation formats
#     if isinstance(observation, tuple):
#         observation = observation[0]
#     else:
#         observation = observation
#     action = agent.act(observation) # your agent here (this currently takes random actions)
#     observation, reward, done, truncated, info = env.step(action)
#     print(done)
#     agent.observe(observation, reward, done)
#     rewards.append(reward)
    
#     if done:
#         observation, info = env.reset() 
# plot_results(args.env, rewards)
# env.close()

num_episodes = 10000
num_runs = 5
all_rewards = []

for run in range(num_runs):
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        else:
            state = state
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            # print(action)
            next_state, reward, done, truncate ,_= env.step(action)
            agent.observe(next_state, reward, done)
            total_reward += reward
            done = done or truncate
            print(done,reward)
            state = next_state
        rewards.append(total_reward)
    all_rewards.append(rewards)
    env.close()

# print(rewards)
window_size = 50
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(env_name, all_rewards):
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    moving_avg_rewards = moving_average(mean_rewards, window_size)
    moving_avg_std = moving_average(std_rewards, window_size)
    ci = 1.96 * moving_avg_std / np.sqrt(num_runs)  # 95% confidence interval

    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg_rewards, label='Moving Average Reward')
    plt.fill_between(range(len(moving_avg_rewards)), moving_avg_rewards - ci, moving_avg_rewards + ci, color='b', alpha=0.2)
    plt.title(f'Rewards for {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

plot_results(args.env, all_rewards)