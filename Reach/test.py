import os                                       # For file and directory operations
import numpy as np                              # For numerical operations, such as calculating the mean
from stable_baselines3 import DDPG              # DDPG (Deep Deterministic Policy Gradient) algorithm
from config import create_env, checkpoint_dir   # Custom functions for environment setup and checkpoint directory
import matplotlib.pyplot as plt                 # For plotting the results
import cv2                                      # For saving frames as a video
import time                                     # Time library for sleep function

def save_video(frames, filename="agent_behavior.mov", fps=30):
    """
    Save a sequence of frames as a video file in .mov format.

    Parameters:
        frames (list): List of frames (images) captured during rendering.
        filename (str): Name of the output video file.
        fps (int): Frames per second for the video.
    """
    # Ensure the 'video_checkpoints' directory exists
    video_dir = "checkpoints_video"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Get frame dimensions from the first frame
    height, width, _ = frames[0].shape

    # Define the codec and create a VideoWriter object for .mov format
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mov files
    video_path = os.path.join(video_dir, filename)  # Save video in 'video_checkpoints' folder
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    # Release the video writer
    video.release()
    print(f"Video saved as {video_path}")

def test_model():
    """
    Test the trained DDPG model using saved checkpoints and visualize the agent's behavior.
    """
    # Step 1: Create the custom environment for evaluation
    env = create_env()

    # Step 2: List all checkpoint files in the specified directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if 'ddpg' in f]
    checkpoint_files.sort()  # Ensure the checkpoints are processed in chronological order

    # Initialize lists to store evaluation metrics
    average_rewards = []  # To track the average rewards for each checkpoint

    # Step 3: Iterate through each checkpoint file to evaluate the model's performance
    for checkpoint_file in checkpoint_files:
        # Construct the full path to the checkpoint file
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

        # Load the DDPG model from the checkpoint file
        model = DDPG.load(checkpoint_path, env=env)

        # Number of episodes to test the model
        num_test_episodes = 10
        rewards = []  # List to store rewards for each test episode
        frames = []   # List to store frames for video rendering

        # Step 4: Run multiple test episodes
        for _ in range(num_test_episodes):
            # Reset the environment and get the initial observation
            obs, info = env.reset()  
            done = False          # Flag to track if the episode is over
            episode_reward = 0    # Accumulator for rewards in the current episode

            # Step 5: Run the episode until it terminates
            while not done:
                # Predict the next action using the trained model
                # Use deterministic actions for evaluation to ensure consistency
                action, _ = model.predict(obs, deterministic=True)

                # Execute the action in the environment and receive feedback
                obs, reward, terminated, truncated, info = env.step(action)

                # Combine the `terminated` and `truncated` flags to determine if the episode is over
                done = terminated or truncated

                # Accumulate the reward obtained in this step
                episode_reward += reward

                # Render the environment and store the frame
                try:
                    frame = env.render()  # Get the rendered frame (without 'mode' argument)
                    frames.append(frame)  # Append the frame to the list
                except TypeError:
                    print("Render failed, skipping frame.")

            # Append the total reward for this episode to the rewards list
            rewards.append(episode_reward)

        # Save the frames as a video for this checkpoint
        video_filename = f"checkpoint_{checkpoint_file}.mov"  # Save as .mov file
        save_video(frames, filename=video_filename)

        # Step 6: Calculate and record the average reward for this checkpoint
        avg_reward = np.mean(rewards)
        average_rewards.append(avg_reward)  # Save the average reward for later analysis

        # Print the checkpoint filename and its corresponding average reward
        print(f"Checkpoint {checkpoint_file}: Average reward over {num_test_episodes} episodes: {avg_reward}")

    # Step 7: Visualize the average rewards over all checkpoints
    plt.figure(figsize=(10, 6))  # Set the size of the figure
    plt.plot(
        range(len(average_rewards)), 
        average_rewards, 
        marker='o', 
        linestyle='-'
    )  # Plot the rewards with markers and lines

    plt.xlabel('Checkpoint Index')       # Label for the x-axis
    plt.ylabel('Average Reward')         # Label for the y-axis
    plt.title('Performance over Checkpoints')  # Title of the plot
    plt.grid()                           # Add a grid for better visualization
    plt.show()                           # Display the plot

# Call the test_model function to start testing
if __name__ == "__main__":
    test_model()
