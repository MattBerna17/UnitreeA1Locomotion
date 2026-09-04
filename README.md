# Unitree A1 Reinforcement-Learning Locomotion

This repository trains a Unitree A1 quadruped in MuJoCo to track a commanded body velocity and follow complex reference trajectories. The custom Gymnasium environment uses the A1 model stored in `unitree_a1/scene.xml`.

> This is a simulation project. A policy trained here must not be deployed on a physical robot without separate safety, state-estimation, latency, actuator, and sim-to-real validation work.

## Repository Layout

| Path | Purpose |
| :--- | :--- |
| `env1.py` | `A1Env`, the custom Gymnasium/MuJoCo task including reward, termination, observation, and action scaling logic. |
| `train.py` | Training entry point supporting PPO, A2C, and SAC algorithms. |
| `free_roam.py` | Script to open the MuJoCo viewer and run a saved policy with correct observation normalization. |
| `evaluate_trajectories.py` | Evaluates the model on predefined trajectories (line, cosine, circle) using a high-level P-controller and logs tracking errors to a CSV. |
| `trajectories.py` | Generates the mathematical reference paths and calculates the cross-track error. |
| `unitree_a1/` | MuJoCo scene files and A1 robot XML assets. |

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
* **PPO & A2C:** Run utilizing 8 parallel environments for 3,000,000 timesteps. Reward normalization is safely enabled as they are on-policy algorithms.
* **SAC:** Runs utilizing 4 parallel environments for 1,000,000 timesteps. Reward normalization is strictly disabled to prevent replay buffer corruption over time. Action space is explicitly bounded to `[-1.0, 1.0]` for SAC compatibility.

To train a model, set the `ALGORITHM` variable inside the script and run:
```bash
python train.py

```

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