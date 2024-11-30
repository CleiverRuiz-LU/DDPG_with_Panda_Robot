# Imports
import gymnasium as gym # Gym defines a framework for virtual environments with states, rewards, actions, and more
import panda_gym        # Builds on top of gym and defines an environment for the panda robot arm

_render_mode = 'human' # Set to 'human' to render the simulation in a window, 
                       # or 'rgb_array' to render the simulation in a notebook
#-------------------
# Section 1
# The RL Loop: reset -> action -> step -> render -> repeat
#-------------------

# Instantiate the panda environment
env = gym.make("PandaReach-v3", render_mode=_render_mode)

# Reset the environment: returns an observation of the environment. observation captures all the information about the simulation
obs, info = env.reset()
print(obs)

# Done is a flag that indicates whether you have reached the max limit of steps or reached a terminal state
done = False

#-------------------
# Simple Controller:
#-------------------
# Given a random desired_goal the robot will move from the current position to the goal. 
# Action Space: in Cartesian coordinates (x,y,z)
while not done:
    
    # Extract current position from obs dict
    current_position = obs["observation"][0:3]
    
    # Extract desired position from obs dict
    desired_position = obs["desired_goal"][0:3]
    
    # Encode the action by computing the difference of goal-actual
    action = 5.0 * (desired_position - current_position)
    
    # Render the robot environment before actually moving
    env.render()
    
    # Pass the action to the robot (the environment). This command calls the robot controllers and moves the robot motors. 
    # The simulator will then return the state, any rewards, done, and extra information. 
    obs, reward, done, truncated, info = env.step(action)

env.close() # Always close the environment at the end to free memory.

#-------------------
# Section 2
# DDPG - Deep Deterministic Policy Gradient
#-------------------
"""
In this section of the code we will train an RL policy and then test it. 
You will notice we always have this two step structure in RL: train and test.
You will want to iteratively train your robot, visualizing and measuring the performance (i.e. how many rewards it accumulates) 
every n-time steps until the robot reaches a satisfactory performance.

The actual algorithm Deep Deterministic Policy Gradient (DDPG) is defined in the stable_baselines3 library.

** Note - DDPG:
See any of these nice overviews of DDPG: 
1) https://spinningup.openai.com/en/latest/algorithms/ddpg.html
2) DDPG and SAC Lecture by Pieter Abeel (https://www.dropbox.com/scl/fi/302ef41a9929yvtedc77l/l5-DDPG-SAC.pdf?rlkey=xc21zgliwfjynjse1je8oo6mx&e=1&dl=0)
3) Pieter Abeel Video on DDPG and SAC   (https://www.youtube.com/watch?v=pg-lKy7JIRk)

** Note - Numpy Arrays:
For this section of the code, we will be using numpy arrays or tensors to define the actions and observations.
You can study these as part of the python notebook we studied early in the semester. 
Numpy arrays behave like matlab matrices and contains various functions to manipulate them.

** Note - Hindsight Experience Replay (HER):
Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., et al. (2017). "Hindsight Experience Replay." In Advances in Neural Information Processing Systems (NeurIPS 2017), pages 5048–5058.

See the following video and slides for an overview of HER:
https://www.youtube.com/watch?v=77xkqEAsHFI

Hindsight Experience Replay (HER) is a reinforcement learning technique designed to improve learning in 
environments with sparse and binary rewards. In such environments, the agent only receives a reward when 
it exactly achieves the goal, which can make learning inefficient due to the rarity of successful episodes.

HER addresses this by allowing the agent to learn from failures. The main idea is to reinterpret unsuccessful 
episodes as successful ones by replacing the original goal with a goal that was actually achieved during the episode. 
This is known as goal relabeling.

Hindsight Experience Replay (HER) is a reinforcement learning technique designed to improve learning in environments 
with sparse and binary rewards. In such environments, the agent only receives a reward when it exactly achieves the goal, 
which can make learning inefficient due to the rarity of successful episodes.

HER addresses this by allowing the agent to learn from failures. The main idea is to reinterpret unsuccessful episodes 
as successful ones by replacing the original goal with a goal that was actually achieved during the episode. This is 
known as **goal relabeling**.

Key aspects of HER:

- **Goal Relabeling**: After an episode, the algorithm replaces the original goal with a different goal that the agent 
    did achieve. This way, the agent learns about how to achieve various goals, using the same experience ("achieved_goals")
- **Experience Replay**: HER works with a replay buffer (memory buffer) that stores the agent's experiences (s,a,r,s',done). 
    When sampling experiences for training, it includes both the original and the relabeled ones.
- **Improved Sample Efficiency**: By learning from every episode, regardless of initial success, HER increases the amount 
    of useful training data, which leads to more efficient learning.

In summary, HER enhances the learning process in challenging environments by enabling the agent to extract meaningful 
    learning signals from episodes that would otherwise be considered failures.

"""

import gymnasium as gym         # Simulated RL environment
import panda_gym                # Panda robot environment definition

import numpy as np              # Linear algebra library

from stable_baselines3 import DDPG, HerReplayBuffer                 # RL algorithm and replay buffer definition
from stable_baselines3.common.noise import NormalActionNoise        # Noise definition for exploration
from sb3_contrib.common.wrappers import TimeFeatureWrapper          # Helps to define the end of an episode
from stable_baselines3.common.callbacks import CheckpointCallback   # Save the model at different periods called checkpoints

import matplotlib.pyplot as plt                                     # Plotting library
import os                                                           # Operating system library. We will  use to get directories in folder

# Kwargs is a "keyword arguments" dictionary that help define options for the algorithm.

# HER Kwargs:
rb_kwargs = {#'online_sampling' : True,                  # HER: sample transitions online
             'goal_selection_strategy' : 'future',     # HER: when you extract a transition achieved in the past, change its goal to one was achieved later in the episode. Teaches the algorithm to achieve future goals from same states. Common strategies include 'future', 'final', and 'episode'. 'future' is the most commonly used.
             'n_sampled_goal' : 4}

# **Deep Neural Network Kwargs:
# In DDPG we will have 2 types of NNs: (i) policy or actor pi(s)->a and (ii) Q-function or critic Q(s,a)

# Actor|Policy Network Output:
#  The actor network takes the state as input and produces an action as the output.
#  Size of the output layer corresponds to the action space of the problem. For example, if your action is a continuous value, 

# The policy NN will learn the optimal policy, while the Q-function NN will learn the optimal value function.
policy_kwargs = {'net_arch' : [512, 512, 512],          # Neural Network Architecture: 3 hidden layers of 512 neurons (input/output not included)
                 'n_critics' : 2}                       # Critic is another word for Q-function. We will create two. Each onen will have a different value. 
                                                        # Because they are optimistic, we will take the min value of the two Q-functions.

# Extract dimensionality of action space from the environment (in this case will be 3 for [x,y,z])
n_actions = env.action_space.shape[0]

# Define the noise for exploration as a Gaussian noise with mean 0 and std 0.1
noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))

# Create the environment and wrap it with the TimeFeatureWrapper
env = gym.make("PandaReach-v3")
env = TimeFeatureWrapper(env)

# Number of steps for training (i.e. 1M)
train_steps = 100000

# Create the DDPG model
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


# Train the DDPG model
# Define a checkpoint callback to save the model every 100,000 steps
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/', name_prefix='ddpg')

# Train the DDPG model with the checkpoint callback
model.learn(train_steps, callback=checkpoint_callback)  # Train the model for 1 million time steps and save every 100,000 steps.

# Save the final trained model
model.save('pick_place/model')  # Save the final trained model to the specified path.

## Testing

# load the model you would like to visualize (checkpoints or final model)
# Define the directory where checkpoints are saved
checkpoint_dir = './checkpoints/'
final_dir = 'pick_place/model'


# Checkpoint files handling logistics
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if 'ddpg' in f]   # List all checkpoint files saved in the checkpoint directory
checkpoint_files.sort()                                                     # Sort the checkpoint files to ensure they are processed in order of training

# Prepare average_rewards list (our performance data) to visualization/plot later
average_rewards = []
frames = []             # Store frames for rendering the animation

# Loop through each checkpoint, load the model, and evaluate it
for checkpoint_file in checkpoint_files:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    
    # Load the model from the checkpoint
    model = DDPG.load(checkpoint_path, env=env)

    # Evaluate the model: Run test episodes and get the rewards
    num_test_episodes = 10
    rewards = []

    for _ in range(num_test_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Predict action using the trained model without exploration noise
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            frame = env.render()
            frames.append(frame)

        rewards.append(episode_reward)

    # Calculate average reward for visualization
    avg_reward = np.mean(rewards)
    average_rewards.append(avg_reward)

    print(f"Checkpoint {checkpoint_file}: Average reward over {num_test_episodes} episodes: {avg_reward}")

## Plot performance
# Plotting the learning progress over checkpoints
plt.figure(figsize=(10, 6))
plt.plot(range(len(average_rewards)), average_rewards, marker='o', linestyle='-')
plt.xlabel('Checkpoint Index')
plt.ylabel('Average Reward')
plt.title('Learning Progress Over Checkpoints')
plt.grid()
plt.show()

env.close()