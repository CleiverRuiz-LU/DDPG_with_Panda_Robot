import gymnasium as gym                                         # Gymnasium for creating and managing reinforcement learning environments
import panda_gym                                                # A library providing Panda robot simulation environments
from sb3_contrib.common.wrappers import TimeFeatureWrapper      # Wrapper to add time features to observations
from stable_baselines3.common.noise import NormalActionNoise    # To add exploration noise
import numpy as np                                              # For numerical operations

# Function to create and configure the environment
def create_env():
    """
    Create the PandaReach-v3 environment and apply necessary wrappers.

    Returns:
        gym.Env: A configured Gymnasium environment.
    """
    # Create the PandaReach-v3 environment
    env = gym.make("PandaReach-v3")
    
    # Apply the TimeFeatureWrapper to include time-related features in the observations
    env = TimeFeatureWrapper(env)
    return env

# Hyperparameters for Hindsight Experience Replay (HER)
rb_kwargs = {
    'goal_selection_strategy': 'future',                                        # Use 'future' strategy to relabel goals from future states
    'n_sampled_goal': 4                                                         # Number of alternative goals to sample per experience
}

# Policy network architecture for the agent
policy_kwargs = {
    'net_arch': [512, 512, 512],                                                # Three fully connected layers with 512 units each
    'n_critics': 2                                                              # Number of critics in the model (used for Q-value estimation)
}

# Create a temporary environment to extract action space properties
env = create_env()                                                              # Instantiate the environment

# Define the number of actions based on the environment's action space
n_actions = env.action_space.shape[0]                                           # Get the dimensionality of the action space

# NormalActionNoise, adds exploration noise for better learning during training
noise = NormalActionNoise(
    mean    =   np.zeros(n_actions),                                            # Mean of the noise (zero-centered)
    sigma   =   0.1 * np.ones(n_actions)                                        # Standard deviation of the noise (controls exploration intensity)
)

# Define directories and paths for saving model checkpoints and the final model
checkpoint_dir = './checkpoints/'                                               # Directory to store periodic model checkpoints
final_model_path = './models/ddpg_model'                                        # Path to save the final trained model

# Total number of training steps
train_steps = 100000                                                            # Number of timesteps to train the agent




# Notes:
# 1. The PandaReach-v3 environment is part of PandaGym and is used for robotic manipulation tasks.
# 2. The `TimeFeatureWrapper` adds time as a feature in the observation to help with non-stationary environments.
# 3. HER (Hindsight Experience Replay) allows the agent to learn even from failures by relabeling goals.
# 4. `NormalActionNoise` encourages exploration by adding Gaussian noise to actions during training.
# 5. The `policy_kwargs` specify the architecture of the neural networks used for policy and value estimation.
# 6. `train_steps` can be adjusted based on the complexity of the task and computational resources.
