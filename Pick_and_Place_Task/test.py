import gymnasium as gym                    
import panda_gym                           
import time       
from config import create_env

def test_model():
    """
    Test 
    """
    # Create the custom environment for evaluation
    env = create_env()
    # Run the simulation loop for 200 steps
    for i in range(10000):
        # Sample a random action from the environment's action space
        action = env.action_space.sample()
        
        # Perform the action in the environment and move one step forward
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Add a small delay to slow down the simulation (e.g., 0.05 seconds)
        time.sleep(0.1)

        # Check if the task has ended (terminated or truncated)
        if terminated or truncated:
            # Reset the environment if the task is done or time limit is reached
            observation, info = env.reset()
            
            # Add a short pause after reset for better visualization
            time.sleep(0.1)

    # Close the environment
    env.close()


# Call the test_model function to start testing
if __name__ == "__main__":
    test_model()
