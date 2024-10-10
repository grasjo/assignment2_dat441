import argparse
import gymnasium as gym
import importlib.util
import matplotlib as plt

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="double_q_learning_agent.py")
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

num_experiments = 1
avg_steps_til_best_state_discovered = 0
avg_rewards = 0

#for i in range(num_experiments):
rewards = []
action_dim = env.action_space.n
state_dim = env.observation_space.n

agent = agentfile.Agent(state_dim, action_dim)
num_steps = 0
episode = 0

observation, info = env.reset()
rewards.append(0)
best_state_discovered = False
avg_rewards = 0
num_episodes = 0

for _ in range(10000000): 
    #env.render()
    action = agent.act(observation) # your agent here (this currently takes random actions)
    observation, reward, done, truncated, info = env.step(action)
    rewards[episode] += reward
    avg_rewards += reward
    agent.observe(observation, reward, done)
    num_steps += 1

    if reward >= 1 and not best_state_discovered:
        best_state_discovered = True
        avg_steps_til_best_state_discovered += num_steps
        print('Number of steps til best state discovered: ', num_steps)
    
    if done:
        num_episodes += 1
        observation, info = env.reset()
        #print('Number of steps for episode', num_steps)
        #print('Average return for agent per episode', avg_reward / num_episodes)
        rewards.append(0)
        episode += 1

avg_rewards /= num_experiments
avg_steps_til_best_state_discovered /= num_experiments
print('Average number of steps til best state discovered:', avg_steps_til_best_state_discovered)
print('Average full rewards for one run:', avg_rewards)

try:
    print('Q:', agent.Q)
except:
    # Double q learning agent
    print('Q1:', agent.Q1)
    print('Q2:', agent.Q2)

env.close()
