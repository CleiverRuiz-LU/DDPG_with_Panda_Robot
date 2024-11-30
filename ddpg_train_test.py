"""
This script trains a reinforcement learning (RL) agent using the Deep Deterministic Policy Gradient (DDPG) algorithm
to solve the PandaReach-v3 task. It includes:
1. Hindsight Experience Replay (HER) for improved learning with sparse rewards.
2. Training and checkpointing of the DDPG model.
3. Visualization of the model's performance during testing.

Dependencies:
- gymnasium
- panda_gym
- numpy
- stable_baselines3
- sb3_contrib
- matplotlib
"""

# Imports
import gymnasium as gym                       # Simulation environment
import panda_gym                              # Panda robot environment
import numpy as np                            # Linear algebra
from stable_baselines3 import DDPG, HerReplayBuffer  # RL algorithm and HER replay buffer
from stable_baselines3.common.noise import NormalActionNoise  # Exploration noise
from sb3_contrib.common.wrappers import TimeFeatureWrapper    # Episode time wrapper
from stable_baselines3.common.callbacks import CheckpointCallback  # Model checkpointing
import matplotlib.pyplot as plt               # For plotting
import os                                     # For handling file paths
import time                                   # Module for introducing delays



env = gym.make("PandaReach-v3")  # PandaReach environment
# Wrap environment with TimeFeatureWrapper to include episode timing
env = TimeFeatureWrapper(env)


# HER Kwargs:
# 'future' strategy changes goals to those achieved later in an episode, teaching agents to achieve future goals.
rb_kwargs = {
    'goal_selection_strategy': 'future',  # Use future goals for relabeling
    'n_sampled_goal': 4                   # Number of relabeled goals per transition
}

# Policy Kwargs:
# Actor (Policy Network) outputs an action given the state.
# Critic (Q-function Network) outputs a value given the state-action pair.
policy_kwargs = {
    'net_arch': [512, 512, 512],  # Three hidden layers with 512 neurons each
    'n_critics': 2                # Two critics for stability
}
# Define noise for exploration
# Gaussian noise with mean 0 and standard deviation 0.1 for each action dimension
n_actions = env.action_space.shape[0]
noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Training parameters
train_steps = 1_000_000  # Number of steps for training

# Create directory for checkpoints
checkpoint_dir = './checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists

# Initialize the DDPG model
# The model uses HER replay buffer and noise for better exploration in continuous action spaces
model = DDPG(
    policy              = "MultiInputPolicy",   # Specifies the type of policy network; "MultiInputPolicy" can handle multiple inputs (e.g., states and observations).
    env                 = env,                  # The environment to train the agent in, which follows OpenAI Gym interface.
    replay_buffer_class = HerReplayBuffer,      # Uses a Hindsight Experience Replay (HER) buffer to improve sample efficiency.
    verbose             = 1,                    # Level of logging; 1 for info messages, 0 for silent, and 2 for debug messages.
    gamma               = 0.95,                 # Discount factor for future rewards; determines how much future rewards are considered (value between 0 and 1).
    batch_size          = 2048,                 # Number of experiences to sample from the replay buffer for each update step.
    buffer_size         = 100000,               # Maximum number of experiences stored in the replay buffer.
    replay_buffer_kwargs= rb_kwargs,            # Additional arguments for customizing the replay buffer, passed as a dictionary.
    learning_rate       = 1e-3,                 # Learning rate for updating the model weights; controls the step size during optimization.
    action_noise        = noise,                # Specifies the noise to be added to actions to promote exploration.
    policy_kwargs       = policy_kwargs         # Additional arguments for customizing the policy, passed as a dictionary.
)


# Checkpoint callback to save the model every 10,000 steps
checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=checkpoint_dir, name_prefix='ddpg')

# Train the model
print("Training DDPG model...")
model.learn(total_timesteps=train_steps, callback=checkpoint_callback)
print("Training complete. Saving final model...")
model.save('pick_place/model')  # Save the final trained model

# Close the environment after training
env.close()

# Testing the trained model
print("Testing the trained model...")

# List all saved checkpoints
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if 'ddpg' in f]
checkpoint_files.sort()  # Ensure checkpoints are processed in training order

average_rewards = []  # To store average rewards for each checkpoint
frames = []  # To store frames for rendering visualization

for checkpoint_file in checkpoint_files:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    model = DDPG.load(checkpoint_path, env=env)  # Load checkpoint

    num_test_episodes = 10  # Number of episodes for evaluation
    rewards = []  # Store rewards for each episode

    # Test the model
    for _ in range(num_test_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Predict action
            obs, reward, done, _ = env.step(action)  # Take the action in the environment
            episode_reward += reward

            frame = env.render()  # Render the environment
            # Introduce a delay to slow down the simulation
            time.sleep(0.2)  # Adjust this value as needed (e.g., 0.1 seconds = 100ms)
            frames.append(frame)  # Store the rendered frame

        rewards.append(episode_reward)  # Store the episode reward

    # Log average reward for this checkpoint
    avg_reward = np.mean(rewards)
    average_rewards.append(avg_reward)
    print(f"Checkpoint {checkpoint_file}: Average reward = {avg_reward}")

# Plot learning progress
plt.figure(figsize=(10, 6))
plt.plot(range(len(average_rewards)), average_rewards, marker='o', linestyle='-')
plt.xlabel('Checkpoint Index')
plt.ylabel('Average Reward')
plt.title('Learning Progress Over Checkpoints')
plt.grid()
plt.show()

# Close the environment after testing
env.close()
