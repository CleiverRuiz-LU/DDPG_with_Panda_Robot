from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from config import (
    create_env,                                                     # Function to initialize and configure the training environment
    train_steps,                                                    # Total number of training steps to run the model
    checkpoint_dir,                                                 # Directory where model checkpoints will be saved
    final_model_path,                                               # Path to save the final trained model
    rb_kwargs,                                                      # Additional arguments for the HER Replay Buffer
    policy_kwargs,                                                  # Additional arguments for the policy (network architecture)
    get_noise                                                       # Function to define the action noise for exploration
)

def train_model():
    """
    Train the DDPG model with Hindsight Experience Replay (HER).
    This function initializes the environment, sets up the DDPG model, configures the noise for exploration,
    and saves periodic checkpoints during training. The final model is saved at the end of training.
    """
    # Create and initialize the environment
    env = create_env()

    # Define the action noise for exploration
    noise = get_noise(env.action_space)             # Generate action noise specific to the environment's action space

    # Initialize the DDPG model
    model = DDPG(
        policy              = "MultiInputPolicy",   # Specifies the type of policy network; "MultiInputPolicy" can handle multiple inputs (e.g., states and observations).
        env                 = env,                  # The environment to train the agent in, which follows OpenAI Gym interface.
        replay_buffer_class = HerReplayBuffer,      # Uses a Hindsight Experience Replay (HER) buffer to improve sample efficiency.
        verbose             = 1,                    # Level of logging; 1 for info messages, 0 for silent, and 2 for debug messages.
        gamma               = 0.95,                 # Discount factor for future rewards; determines how much future rewards are considered (value between 0 and 1).
        batch_size          = 2048,                 # Number of experiences to sample from the replay buffer for each update step.
        buffer_size         = 100_000,              # Maximum number of experiences stored in the replay buffer.
        replay_buffer_kwargs= rb_kwargs,            # Additional arguments for customizing the replay buffer, passed as a dictionary.
        learning_rate       = 1e-3,                 # Learning rate for updating the model weights; controls the step size during optimization.
        action_noise        = noise,                # Specifies the noise to be added to actions to promote exploration.
        policy_kwargs       = policy_kwargs         # Additional arguments for customizing the policy, passed as a dictionary.
    )


    # Set up a checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq   =   10_000,                                     # Save the model every 10,000 timesteps
        save_path   =   checkpoint_dir,                             # Directory to save checkpoints
        name_prefix =   'ddpg'                                      # Prefix for checkpoint filenames (e.g., ddpg_10000_steps.zip)
    )

    # Train the model
    model.learn(
        total_timesteps =   train_steps,                            # Total training steps (defined in config)
        callback        =   checkpoint_callback                     # Save periodic checkpoints
    )

    #  Save the final trained model
    model.save(final_model_path)
    print(f"Training complete. Final model saved at {final_model_path}")

if __name__ == "__main__":
    # Entry point of the script
    print("Starting training...")
    train_model()
