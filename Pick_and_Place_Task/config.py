import gymnasium as gym                                         # Gymnasium for creating and managing reinforcement learning environments
import panda_gym                                                # A library providing Panda robot simulation environments
from sb3_contrib.common.wrappers import TimeFeatureWrapper      # Wrapper to add time features to observations
from stable_baselines3.common.noise import NormalActionNoise    # Adds exploration noise for continuous action spaces
import numpy as np                                              # For numerical operations


# Function to create and configure the environment
def create_env():
    """
    Creates an environment for simulating Franka Emika Robotics Arm tasks.

    Available tasks (env_name):
        - PandaReach-v3
        - PandaPush-v3
        - PandaSlide-v3
        - PandaPickAndPlace-v3
        - PandaStack-v3
        - PandaFlip-v3

    Render modes:
        - 'human'      : Renders the simulation in a window for visualization.
        - 'rgb_array'  : Returns an RGB array for off-screen rendering.

    Returns:
        gym.Env: A configured Gymnasium environment instance.
    """
    # Initialize the PandaPickAndPlace-v3 environment with human visualization
    env = gym.make("PandaPickAndPlace-v3", render_mode="human")
    env = TimeFeatureWrapper(env)

    # Reset the environment to its initial state and retrieve the initial observation
    env.reset()
    
    
        
    return env

# Global variables for training configuration
train_steps = 1_000_000    
checkpoint_dir = './checkpoints_Pick_and_Place/'
final_model_path = './models/ddpg_model'

# Hyperparameters for Hindsight Experience Replay (HER)
rb_kwargs = {
    'goal_selection_strategy': 'future',  # Use 'future' strategy to relabel goals
    'n_sampled_goal': 4                   # Number of alternative goals to sample
}

# Policy network architecture
policy_kwargs = {
    'net_arch': [512, 512, 512],          # Fully connected layers with 512 units
    'n_critics': 2                        # Number of critics for Q-value estimation
}

# Function to generate NormalActionNoise for exploration in reinforcement learning
def get_noise(action_space):
    """
    Generates NormalActionNoise for a given action space to encourage exploration.
    """
    # Determine the number of actions based on the shape of the action space
    n_actions = action_space.shape[0]
    
    # Create and return NormalActionNoise with:
    return NormalActionNoise(
        mean    =   np.zeros(n_actions),        # Zero mean for all actions
        sigma   =   0.1 * np.ones(n_actions)    # Standard deviation of 0.1 for all actions
    )
