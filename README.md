# DRLND-Udacity-p2-Continuous-Control

## Description 
For this project, we train a double-jointed arm agent to follow a target location.

## Problem Statement 
A reward of +0.1 is provided for each step that the agent's hands is in the goal location.
Thus, the goal of the agent is to maintain its position at
the target location for as many
steps as possible.

The observation space consists of 33 variables corresponding to position, 
rotation, velocity, and angular velocities of the arm. 
Each action is a vector with four numbers, corresponding to torque
applicable to two joints. Every 
entry in the action vector should be a number between -1 and 1. 

The task is episodic, with 1000 timesteps per episode. In order to solve
the environment, the agent must get an average score of +30 over 100 consecutive
episodes.

## Files 
- `Continuous_Control.ipynb`: Notebook with the code 
- `critic.pth`: Critic trained model
- `actor.pth`: Actor trained model 
- `report.md`: Technical report
## Dependencies
To be able to run this code, you will need an environment with Python 3 and 
the dependencies are listed in the `requirements.txt` file so that you can install them
using the following command: 
```
pip install requirements.txt
``` 

Furthermore, you need to download the environment from one of the links below. You need only to select
the environment that matches your operating system:
- Linux : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- MAC OSX : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Running
Run the cells in the notebook `Continuous_Control.ipynb` to train an agent that solves our required
task of moving the double-jointed arm.

