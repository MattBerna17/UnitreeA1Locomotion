# Unitree A1 Reinforcement-Learning Locomotion

This repository trains a Unitree A1 quadruped in MuJoCo to track a commanded
body velocity. The custom Gymnasium environment uses the A1 model stored in
`unitree_a1/scene.xml`; that scene includes `a1.xml` and its mesh assets.

> This is a simulation project. A policy trained here must not be deployed on a
> physical robot without separate safety, state-estimation, latency, actuator,
> and sim-to-real validation work.

## Repository layout

| Path | Purpose |
| --- | --- |
| `env1.py` | `A1Env`, the custom Gymnasium/MuJoCo task and its reward, termination, observation, and action logic. |
| `train.py` | PPO training entry point using 8 parallel environments and observation/reward normalization. |
| `main.py` | Opens the MuJoCo viewer and runs a saved PPO policy. |
| `unitree_a1/scene.xml` | MuJoCo scene with the floor; includes the A1 robot XML. |
| `unitree_a1/a1.xml` | A1 model, 12 position actuators, joint limits, and the `home` keyframe. |
| `environment.yml` | Portable Conda environment definition. |
| `requirements.txt` | Equivalent pip dependency list. |

## Environment design

The A1 has a floating base and 12 position actuators: abduction, hip, and knee
for each leg in FR, FL, RR, RL order. Therefore an action is a vector of 12
joint-position targets, **not a torque vector**.

Each policy decision advances 10 MuJoCo steps. With the model's default
0.002-second simulator timestep this corresponds to a 50 Hz control rate. An
episode lasts up to 15 seconds and ends early when trunk height, roll, or pitch
is outside the configured healthy range.

The 48-value observation is composed of base linear velocity, base angular
velocity, gravity projected into the trunk frame, the desired velocity command,
joint positions relative to the home pose, joint velocities, and the preceding
action.

The current reward combines linear and angular velocity tracking, healthy state,
diagonal-contact (trot), and foot-air-time terms. It subtracts costs for actuator
effort, vertical and angular velocity, abrupt action changes, joint limits,
joint acceleration, base orientation, body height, flight phases, and foot slip.

## Prerequisites

- Conda or Miniforge is recommended; Python 3.10 is used by the project.
- A working OpenGL installation is needed to open the interactive MuJoCo viewer.
- Run commands from the repository root, so `unitree_a1/scene.xml` resolves
  correctly.

## Installation

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate a1-rl
```

To update an existing environment after a dependency change:

```bash
conda env update -n a1-rl -f environment.yml --prune
```

### Option B — pip and a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate              # macOS/Linux
# .venv\Scripts\activate               # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train

Before running the current `train.py`, make this one-line correction. `A1Env`
already supplies the XML model path internally, and Gymnasium's `MujocoEnv`
does not accept `xml_path` as a keyword argument.

```python
# Replace this:
env = A1Env(xml_path="./unitree_a1/scene.xml", render_mode=None)

# With this:
env = A1Env(render_mode=None)
```

Then run:

```bash
python train.py
```

The script first runs Gymnasium's environment checker, trains PPO for
10,000,000 transitions, and writes:

- `ppo_a1.zip` — learned PPO policy;
- `vecnormalize_a1.pkl` — normalization statistics required at evaluation time;
- `tensorboard_logs/` — TensorBoard event files.

Monitor the run in a second terminal:

```bash
conda activate a1-rl
tensorboard --logdir tensorboard_logs
```

Open the displayed local URL in a browser.

## Run a trained policy

`main.py` loads `ppo_a1.zip` and opens the MuJoCo viewer:

```bash
python main.py
```

The evaluation environment must use the same observation normalization saved in
`vecnormalize_a1.pkl`. The current `main.py` loads the PPO file directly but
does **not** restore `VecNormalize`; this gives the policy observations on a
different scale from training. Update it to load the saved normalization wrapper
before treating the viewer behaviour as a valid evaluation result.

## Reproducibility and repository hygiene

- Commit source files, XML assets, `environment.yml`, `requirements.txt`, this
  README, and small configuration files.
- Do not commit `.venv/`, `__pycache__/`, TensorBoard logs, videos, or large
  checkpoints. Keep a selected released checkpoint only if its size is
  acceptable, otherwise use Git LFS or a release/archive store.
- When dependencies change, edit the short dependency lists deliberately; do
  not commit a full OS-specific `conda env export` unless the exact machine
  build is required for archival reproducibility.

## Important training notes

- The desired forward velocity is currently fixed at 2.5 m/s in `env1.py`.
  This is a demanding initial target. For gait bootstrapping, begin at a lower
  speed (for example 0.2–0.8 m/s) and broaden the command range only after the
  robot walks stably.
- Do not use joint actions or the home `qpos` vector as a distance reward. True
  forward progress is the change in trunk world `x` position; velocity tracking
  is usually the more controlled objective for a commanded walking task.
- The reward weights are hyperparameters. Log every component and compare their
  magnitudes before changing several terms at once.
