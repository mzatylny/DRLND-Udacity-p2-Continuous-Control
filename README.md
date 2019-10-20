# DRLND-Udacity-p2-Continuous-Control

## Description 
For this project, we train a double-jointed arm agent to follow a target location.

<p align="center">
	<img src="images/reacher_gif.gif" width=50% height=50%>
</p>

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
- `Continuous_Control.ipynb`: Notebook used to control and train the agent 
- `ddpg_agent.py`: Create an Agent class that interacts with and learns from the environment 
- `model.py`: Actor and Critic classes  
- `config.json`: Configuration file to store variables and paths
- `utils.py`: Helper functions 
- `report.pdf`: Technical report
