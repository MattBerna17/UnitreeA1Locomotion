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
| `plot_trajectories.py` | Generates analytical dashboards comparing algorithms based on the CSV data. |
| `trajectories.py` | Generates the mathematical reference paths and calculates the cross-track error using orthogonal projection. |

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

* The A1 utilizes position actuators rather than pure torque control. Position control explicitly delegates high-frequency PD torque calculations to the lower-level solver, which mitigates the *sim-to-real* reality gap by protecting the physical robot from chaotic Out-of-Distribution network outputs.


* The control frequency is set to 50 Hz, achieved by skipping 10 MuJoCo frames per policy step (0.02s dt).


* The network output relies on Residual Control: actions are scaled by a factor of `0.25` and added to the robot's default joint positions.


* The target velocity is dynamically sampled between `[0.3, -0.0, -1.0]` and `[0.3, 0.0, 1.0]` representing `(vx, vy, wz)`, allowing the robot to learn how to steer.


* The final step reward is clamped at zero using `max(rewards - penalties, 0.0)`. This Reward Shaping strategy prevents the *early termination anomaly* (policy suicide), ensuring the agent is incentivized to explore suboptimal gaits rather than intentionally crashing the robot to avoid accumulating negative scores.



## Training

The `train.py` script supports multiple reinforcement learning architectures, highlighting the trade-off between sample efficiency and computational overhead:

* **PPO:** Configured with a batch size of 64 and 2048 n_steps. It is highly stable and computationally fast (wall-clock time), leveraging large rollout buffers. Reward normalization is safely enabled via `VecNormalize` as it relies on on-policy data collection.


* **SAC:** Configured with `ent_coef="auto"` to maximize entropy for robust gait exploration, and performs 8 gradient steps per environment step. Reward normalization is strictly disabled to prevent replay buffer non-stationarity over time.



## Evaluation & Trajectory Tracking

### Free Roam

`free_roam.py` loads the saved model alongside its `VecNormalize` statistics (for PPO). This guarantees the policy receives observations on the exact same mathematical scale used during training.

### Closed-Loop Navigation

`evaluate_trajectories.py` tests the robot's ability to navigate continuous curves:

* It utilizes a high-level P-controller (`KP_HEADING = 2.0`) to calculate the angular velocity command (`wz`) based on the heading error towards a lookahead point on the reference trajectory.


* **High-Precision Tracking Metric:** Cross-track error is evaluated by calculating the orthogonal projection of the robot's position onto the continuous trajectory segments. This eliminates spatial aliasing (sawtooth noise) caused by discrete point nearest-neighbor algorithms, providing a true sub-millimeter Mean Absolute Error (MAE) evaluation.


* The angular command is saturated at `1.0` rad/s to respect the bounds of the training domain.


* The evaluation automatically terminates when the robot reaches within 20 cm of the final trajectory point.


* The script generates a comprehensive 2x2 joint angle plot separating the behavior of the FR, FL, RR, and RL legs during locomotion.



## Recordings & Results

Below are the visual recordings of the policies attempting the trajectory tracking task. The differences in gait and stability reflect the divergent learning strategies of on-policy versus off-policy algorithms.

**PPO Trajectory Tracking**

*(Displays a stable, rhythmic trot with excellent cross-track adherence)*
[![PPO Trajectory Tracking](https://img.youtube.com/vi/8_3IQFpWdJk/0.jpg)](https://youtu.be/fdHoaDwx9yg)

**SAC Trajectory Tracking**

*(Displays an asymmetric, high-frequency gait pattern)*
[![SAC Trajectory Tracking](https://img.youtube.com/vi/EeV1KnvGf2c/0.jpg)](https://youtu.be/RwWdwD8RRy0)

```