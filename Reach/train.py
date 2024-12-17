from stable_baselines3 import DDPG, HerReplayBuffer                     # DDPG algorithm and HER replay buffer for reinforcement learning
from stable_baselines3.common.callbacks import CheckpointCallback       # Callback to save model checkpoints during training
from config import (
    create_env,                         # Function to create the training environment
    rb_kwargs,                          # Hindsight Experience Replay (HER) replay buffer configuration
    policy_kwargs,                      # Policy network configuration
    noise,                              # Action noise for exploration
    train_steps,                        # Total number of training steps
    checkpoint_dir,                     # Directory to save model checkpoints
    final_model_path                    # Path to save the final trained model
)


def train_model():
    """
    Train the DDPG model with Hindsight Experience Replay (HER) 
    and save periodic checkpoints during training.
    """
    # Create the custom environment for training
    env = create_env()

    # Initialize the DDPG model
    model = DDPG(
        policy                  =   "MultiInputPolicy",     # Policy type, suitable for environments with multiple input spaces
        env                     =   env,                    # Training environment
        replay_buffer_class     =   HerReplayBuffer,        # Use HER for relabeling goals in the replay buffer
        verbose                 =   1,                      # Set verbosity to 1 to display training progress in the console
        gamma                   =   0.95,                   # Discount factor for future rewards
        batch_size              =   2048,                   # Size of the training batches
        buffer_size             =   100000,                 # Size of the replay buffer (number of transitions stored)
        replay_buffer_kwargs    =   rb_kwargs,              # HER-specific replay buffer configuration
        learning_rate           =   1e-3,                   # Learning rate for the optimizer
        action_noise            =   noise,                  # Add Gaussian noise for exploration
        policy_kwargs           =   policy_kwargs           # Neural network architecture for the policy and value functions
    )

    # Define a callback to save model checkpoints periodically
    checkpoint_callback = CheckpointCallback(
        save_freq   =   10000,                              # Save a checkpoint every 10,000 timesteps
        save_path   =   checkpoint_dir,                     # Directory to save the checkpoints
        name_prefix =    'ddpg'                             # Prefix for the checkpoint filenames
    )

    # Start training the model
    model.learn(
        total_timesteps =   train_steps,                    # Total number of timesteps to train the model
        callback        =   checkpoint_callback             # Use the checkpoint callback during training
    )

    # Save the final trained model to the specified path
    model.save(final_model_path)
    print(f"Training complete. Final model saved at {final_model_path}")

# Main
if __name__ == "__main__":
    print("Starting training...")                           # Indicate that the training process is starting
    train_model()                                           # Call the training function
