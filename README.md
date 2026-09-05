# Unitree A1 Reinforcement-Learning Locomotion

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-20124d?style=for-the-badge)
![MuJoCo](https://img.shields.io/badge/MuJoCo-000000?style=for-the-badge)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1D1D1D?style=for-the-badge)
![Unitree A1](https://img.shields.io/badge/Unitree_A1-FF6600?style=for-the-badge)

This repository trains a Unitree A1 quadruped in MuJoCo to track a commanded body velocity and follow complex reference trajectories. The custom Gymnasium environment uses the A1 model stored in `unitree_a1/scene.xml`.

> This is a simulation project. A policy trained here must not be deployed on a physical robot without separate safety, state-estimation, latency, actuator, and sim-to-real validation work.

## Repository Layout

| Path | Purpose |
| :--- | :--- |
| `env1.py` | `A1Env`, the custom Gymnasium/MuJoCo task including reward, termination, observation, and action scaling logic. |
| `train.py` | Training entry point supporting PPO and SAC algorithms. |
| `free_roam.py` | Script to open the MuJoCo viewer and run a saved policy with correct observation normalization. |
| `evaluate_trajectories.py` | Evaluates the model on predefined trajectories (line, cosine, circle) using a high-level P-controller and logs tracking errors to a CSV. |
| `plot_trajectories.py` | Generates a 1x3 analytical dashboard comparing algorithms based on the CSV data. |
| `trajectories.py` | Generates the mathematical reference paths and calculates the cross-track error. |
| `unitree_a1/` | MuJoCo scene files and A1 robot XML assets. |

## Quick Start & Execution

Follow these sequential steps to set up the environment, train, and evaluate the models:

```bash
# 1. Activate your virtual environment (Conda or venv)
conda activate a1-rl

# 2. Install the required dependencies
pip install -r requirements.txt

# 3. Train the models (Optional, pre-trained models can be used)
python train.py

# 4. Evaluate the trajectories in the MuJoCo viewer 
# (Note: mjpython is required on macOS to render the viewer correctly)
mjpython evaluate_trajectories.py

# 5. Generate the comparison dashboards and joint angle plots
python plot_trajectories.py

```

## Environment Design

* The A1 has a floating base and 12 position actuators.


* Each policy decision advances 10 MuJoCo frames, resulting in a 50 Hz control rate (0.02s dt).


* Episodes last up to 15 seconds.


* The target velocity is dynamically sampled between `[0.3, -0.0, -1.0]` and `[0.3, 0.0, 1.0]` representing `(vx, vy, wz)`, allowing the robot to learn how to steer.


* The 48-value observation includes base linear/angular velocity, projected gravity, desired velocity, joint positions relative to home, joint velocities, and the preceding action.


* The reward combines linear and angular velocity tracking, healthy state, and foot-air-time terms.


* Costs penalize actuator torque, vertical/angular velocity, abrupt action changes, joint limits, joint acceleration, base orientation, body height, flight phases, foot slip, and collisions.



## Training

The `train.py` script has been expanded to support multiple reinforcement learning architectures:

* **PPO:** Run utilizing 8 parallel environments for 3,000,000 timesteps. Reward normalization is safely enabled as they are on-policy algorithms.


* **SAC:** Runs utilizing 4 parallel environments for 1,000,000 timesteps. Reward normalization is strictly disabled to prevent replay buffer corruption over time. Action space is explicitly bounded to `[-1.0, 1.0]` for SAC compatibility.



## Evaluation & Trajectory Tracking

### Free Roam

`free_roam.py` loads the saved model (`.zip`) alongside its `VecNormalize` statistics (`.pkl`). This guarantees the policy receives observations on the exact same mathematical scale used during training.

### Closed-Loop Navigation

`evaluate_trajectories.py` tests the robot's ability to navigate continuous curves:

* It utilizes a high-level P-controller to calculate the angular velocity command (`wz`) based on the heading error towards a lookahead point on the reference trajectory.


* The angular command is saturated at `1.0` rad/s to respect the bounds of the training domain.


* A green capsule rendering is injected directly into the MuJoCo viewer to visualize the path.


* The script automatically stops when the robot is within 15 cm of the final trajectory point.


* Detailed tracking metrics (cross-track error, physical positions, commands) are exported to `trajectory_tracking_results.csv`.



## Recordings & Results

Below are the visual recordings of the policies attempting the trajectory tracking task. The differences in gait and stability reflect the divergent learning strategies of on-policy versus off-policy algorithms.

**PPO Trajectory Tracking**  
*(Displays a stable, rhythmic trot with excellent cross-track adherence)*  
[![PPO Trajectory Tracking](https://img.youtube.com/vi/8_3IQFpWdJk/0.jpg)](https://youtu.be/8_3IQFpWdJk)

**SAC Trajectory Tracking**  
*(Displays an asymmetric, high-frequency gait pattern)*  
[![SAC Trajectory Tracking](https://img.youtube.com/vi/EeV1KnvGf2c/0.jpg)](https://youtu.be/EeV1KnvGf2c)