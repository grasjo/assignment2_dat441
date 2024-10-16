import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
parser.add_argument("--init_strategy", type=str, help="Initialization strategy", default="ones")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)




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





# all_rewards = []
# num_runs = 5
# for run in range(num_runs):
#     rewards = []
#     episode = 0
#     agent = agentfile.Agent(state_dim, action_dim, terminal_states=terminal_states, init_strategy=args.init_strategy)
#     observation, info = env.reset()
#     # for _ in range(200):

#     #     action = agent.act(observation) # your agent here (this currently takes random actions)
#     #     observation, reward, done, truncated, info = env.step(action)
#     #     rewards.append(reward)
#     #     agent.observe(observation, reward, done)
        
#     #     if done:
#     #         observation, info = env.reset()
#     #         episode += 1
#     # all_rewards.append(rewards)

#     while episode < 100:
        
#         observation, info = env.reset()
#         done = False
#         while not done:
#             action = agent.act(observation) # your agent here (this currently takes random actions)
#             observation, reward, done, truncated, info = env.step(action)
#             agent.observe(observation, reward, done)
#         episode += 1
#         rewards.append(reward)
#     all_rewards.append(rewards)
        
num_runs = 5
all_rewards = []
for run in range(num_runs):
    rewards = []
    episode = 0
    agent = agentfile.Agent(state_dim, action_dim, terminal_states=terminal_states, init_strategy=args.init_strategy)
    observation, info = env.reset()
    while episode < 10000:
        done = False
        while not done:
            action = agent.act(observation) # your agent here (this currently takes random actions)
            observation, reward, done, truncated, info = env.step(action)
            agent.observe(observation, reward, done)
        observation, info = env.reset()
        episode += 1
        rewards.append(reward)
    print(len(rewards))
    all_rewards.append(rewards)

# try:
#     print(np.argmax(agent.Q, axis=1))
# except:
#     print(np.argmax(agent.Q1 + agent.Q2, axis=1))
env.close()

window_size = 50
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(env_name, all_rewards):
    mean_rewards = np.mean(all_rewards, axis=0)
    print(len(mean_rewards))
    std_rewards = np.std(all_rewards, axis=0)
    moving_avg_rewards = moving_average(mean_rewards, window_size)
    moving_avg_std = moving_average(std_rewards, window_size)
    ci = 1.96 * moving_avg_std / np.sqrt(num_runs)  # 95% confidence interval
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg_rewards, label='Moving Average Reward')
    plt.fill_between(range(len(moving_avg_rewards)), moving_avg_rewards - ci, moving_avg_rewards + ci, color='b', alpha=0.2)
    plt.title(f'Rewards for {env_name}, Agent: SARSA, {args.init_strategy} initialization')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

plot_results(args.env, all_rewards)
