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
from stable_baselines3 import DDPG            # RL algorithm
from stable_baselines3.common.noise import NormalActionNoise  # Exploration noise
from sb3_contrib.common.wrappers import TimeFeatureWrapper    # Episode time wrapper
from stable_baselines3.common.callbacks import CheckpointCallback  # Model checkpointing
import matplotlib.pyplot as plt               # For plotting
import os                                     # For handling file paths

# Define HER replay buffer arguments
rb_kwargs = {
    'goal_selection_strategy': 'future',  # Use future goals for relabeling
    'n_sampled_goal': 4                   # Number of relabeled goals per transition
}

# Define policy network arguments (shared by Actor and Critic networks)
policy_kwargs = {
    'net_arch': [512, 512, 512],  # Three hidden layers with 512 neurons each
    'n_critics': 2                # Two critics for stability
}

# Define noise for exploration (Gaussian noise with mean 0 and std 0.1)
env = gym.make("PandaReach-v3")
n_actions = env.action_space.shape[0]
noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Wrap environment with TimeFeatureWrapper to handle episode timing
env = TimeFeatureWrapper(env)

# Training parameters
train_steps = 1_000_000  # Number of steps for training
checkpoint_dir = './checkpoints/'  # Directory to save checkpoints

# Initialize the DDPG model
model = DDPG(
    policy="MultiInputPolicy",
    env=env,
    replay_buffer_class="HerReplayBuffer",
    replay_buffer_kwargs=rb_kwargs,
    policy_kwargs=policy_kwargs,
    action_noise=noise,
    batch_size=2048,
    buffer_size=100_000,
    gamma=0.95,  # Discount factor
    learning_rate=1e-3,  # Learning rate
    verbose=1  # Logging level
)

# Checkpoint callback to save the model every 10,000 steps
checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=checkpoint_dir, name_prefix='ddpg')

# Train the model
print("Training DDPG model...")
model.learn(total_timesteps=train_steps, callback=checkpoint_callback)
print("Training complete. Saving final model...")
model.save('pick_place/model')  # Save final trained model

# Close the environment after training
env.close()

# Testing the trained model
print("Testing the trained model...")

# List all saved checkpoints
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if 'ddpg' in f]
checkpoint_files.sort()  # Ensure checkpoints are processed in training order

average_rewards = []  # To store average rewards for each checkpoint

for checkpoint_file in checkpoint_files:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    model = DDPG.load(checkpoint_path, env=env)  # Load checkpoint

    num_test_episodes = 10
    rewards = []

    # Test the model
    for _ in range(num_test_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Predict action
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

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
