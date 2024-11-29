# **DDPG Panda Robot Environment Setup**

This repository provides step-by-step instructions for setting up the environment required to run Deep Deterministic Policy Gradient (DDPG) algorithms using the Panda robot simulation provided by `panda-gym` and `stable-baselines3`. This setup is ideal for researchers and developers interested in reinforcement learning with robotic manipulators.

# **Table of Contents**

1. [Prerequisites](#prerequisites)
2. [Installation Instructions](#installation-instructions)
   - [1. Install Conda](#1-install-conda)
   - [2. Create a Conda Environment](#2-create-a-conda-environment)
   - [3. Activate the Environment](#3-activate-the-environment)
   - [4. Install Required Packages](#4-install-required-packages)
   - [5. Verify Installed Packages](#5-verify-installed-packages)
3. [Additional Resources](#additional-resources)


## **Prerequisites**

- **Operating System:** macOS, Windows, Linux
- **Python Version:** 3.9
- **Conda:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://docs.anaconda.com/anaconda/)

Ensure you have administrative privileges to install software on your system


## **Installation Instructions**

### **1. Install Conda**

If you haven't already, install **Miniconda** or **Anaconda**. Miniconda is recommended for a lightweight installation.

- **Download Anaconda or Miniconda:**
  - Visit the [Miniconda Installation Page](https://docs.anaconda.com/miniconda/install/).
  - Choose the installer for your OS.

### **2. Create a Conda Environment**

Conda allows you to create isolated environments with specific dependencies, preventing conflicts between projects.

- **Create the environment:**

  ```bash
  conda create --name ddpg_panda_env python=3.9
  ```

  - **Explanation:**
    - `conda create`: Command to create a new environment.
    - `--name ddpg_panda_env`: Names the environment `ddpg_panda_env`.
    - `python=3.9`: Specifies Python version 3.9.

- **Note:** Both `panda-gym` and `stable-baselines3` require **Python 3.9** to run correctly.

### **3. Activate the Environment**

Activate the newly created environment to ensure all packages are installed within it.

- **Activate the environment:**

  ```bash
  conda activate ddpg_panda_env
  ```

  - **Verification:**
    - Your terminal prompt should now display `(ddpg_panda_env)` indicating the environment is active.

- **Deactivate when done:**

  ```bash
  conda deactivate
  ```

### **4. Install Required Packages**

With the environment activated, install the necessary packages using `pip`.

- **Install packages:**

  ```bash
  pip install gymnasium         # Simulated environments using PyBullet as the physics engine
  pip install panda-gym         # Panda robot arm simulation with reinforcement learning interface
  pip install stable-baselines3 # Reinforcement Learning algorithms (e.g., DDPG, SAC)
  pip install sb3_contrib       # Additional tools and algorithms for stable-baselines3
  ```

  - **Package Descriptions:**
    - **gymnasium:** A toolkit for developing and comparing reinforcement learning algorithms.
    - **panda-gym:** Simulated environments for the Franka Emika Panda robot.
    - **stable-baselines3:** Set of reliable implementations of RL algorithms in PyTorch.
    - **sb3_contrib:** Community contributions to stable-baselines3.

- **Sample Installation Output:**

  ![Package Installation](/doc/image/conda_init_ddpg_robot.png)

### **5. Verify Installed Packages**

Ensure that all packages are correctly installed.

- **List installed packages:**

  ```bash
  conda list
  ```

- **Sample Output:**

  ![Conda List](/doc/image/conda_list_ddpg_robot.png)

  - **Verification:**
    - Check that `gymnasium`, `panda-gym`, `stable-baselines3`, and `sb3_contrib` are listed with their respective versions.

### **6. Interpreter Issues**
If your IDE such as Visual Stuido Code is showing issues even when all pakcages and edepnedaces have been installed. insure that the correct interpote has be selced. this cna b done by naviagitng to command pallets selection "Python: Select Interpreter" and slecting the correct pythong interoper wiht oyur created enviorment

  ![Interpreter Solution](/doc/image/python_interp.gif)




## **Additional Resources**

- **Panda-Gym Documentation:** [Panda-Gym Docs](https://panda-gym.readthedocs.io/en/latest/)
  - Explore different environments, customization options, and usage examples.

- **Stable-Baselines3 DDPG Documentation:** [Stable-Baselines3 DDPG](https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html)
  - Learn about the DDPG algorithm implementation, parameters, and best practices.

