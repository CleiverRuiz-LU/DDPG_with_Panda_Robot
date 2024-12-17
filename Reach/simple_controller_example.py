"""
This script demonstrates the RL loop for controlling a Panda robot arm using a simple controller.
The robot's task is to move its end-effector to a desired goal position.

Dependencies:
- gymnasium
- panda_gym

Rendering Modes:
- 'human': Renders the simulation in a window.
- 'rgb_array': Renders the simulation as an image, useful for notebooks.
"""

# Imports
import gymnasium as gym  # Gym for defining and simulating virtual environments
import panda_gym         # Panda robot environment for Gym
import time              # Module for introducing delays

# Rendering mode: Options are 'human' or 'rgb_array'
_render_mode = 'human'

# 1. Instantiate the Panda environment
env = gym.make("PandaPickAndPlace-v3", render_mode=_render_mode)

# 2. Reset the environment to start: Returns the initial observation and environment info
obs, info = env.reset()
print(f"Initial Observation: {obs}")

# Flag to track the end of an episode
done = False

# 3. Simple Controller: Move the robot from the current position to a desired goal
while not done:
    # Extract the current position of the robot's end-effector
    current_position = obs["observation"][0:3]

    # Extract the desired goal position
    desired_position = obs["desired_goal"][0:3]

    # Compute the action as the difference between the goal and current position
    # Scaled by a factor to ensure smooth movement
    action = 5.0 * (desired_position - current_position)

    # Render the robot's environment (visualize the simulation)
    env.render()

    # Introduce a delay to slow down the simulation
    time.sleep(0.5)  # Adjust this value as needed (e.g., 0.1 seconds = 100ms)

    # Step the environment with the computed action:
    # Returns the next state, reward, done flag, truncated flag, and additional info
    obs, reward, done, truncated, info = env.step(action)

# Always close the environment to release resources
env.close()
