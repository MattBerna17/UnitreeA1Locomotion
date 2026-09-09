# evaluates rl models on tracking reference trajectories
# saves cross track error to csv
# calculates and injects angular velocity command
# generates joint angles plots
# usage python evaluate_trajectories.py

import csv
import time
import numpy as np
import mujoco
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg') # force non interactive backend
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env1 import A1Env
from trajectories import TRAJECTORIES, closest_point_on_trajectory

# configuration
MODELS = {
    "PPO": (PPO, "ppo_a1_trajectories_seed_15.zip", "ppo_vecnormalize_a1_trajectories_seed_15.pkl"),
    "SAC": (SAC, "sac_a1_trajectories_seed_15.zip", None),
}

RENDER = True          
MAX_STEPS = 1_000_000
FORWARD_SPEED = 0.3    

LOOKAHEAD_POINTS = 8   
KP_HEADING = 2.0
WZ_MAX = 1.0           

OUTPUT_CSV = "trajectory_tracking_results_.csv"
TRAJECTORY_Z = 0.02    

# visualization
def draw_trajectory(viewer, trajectory_xy, z=TRAJECTORY_Z):
    scn = viewer.user_scn
    scn.ngeom = 0
    step = max(1, len(trajectory_xy) // 300)  
    for i in range(0, len(trajectory_xy) - step, step):
        p1 = np.array([trajectory_xy[i, 0], trajectory_xy[i, 1], z])
        p2 = np.array([trajectory_xy[i + step, 0], trajectory_xy[i + step, 1], z])
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008, p1, p2
        )
        geom.rgba[:] = [0.0, 1.0, 0.0, 1.0]
        scn.ngeom += 1

# steering control
def heading_command(position_xy, yaw, trajectory_xy, closest_idx):
    target_idx = min(closest_idx + LOOKAHEAD_POINTS, len(trajectory_xy) - 1)
    target = trajectory_xy[target_idx]

    desired_yaw = np.arctan2(target[1] - position_xy[1], target[0] - position_xy[0])
    heading_error = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))

    wz = np.clip(KP_HEADING * heading_error, -WZ_MAX, WZ_MAX)
    return wz

# tracking episode execution
def run_trajectory_episode(algo_name, model, vec_env, raw_env, traj_name, trajectory_xy, viewer=None):
    obs = vec_env.reset()

    raw_env.data.qpos[0:2] = trajectory_xy[0]
    mujoco.mj_forward(raw_env.model, raw_env.data)
    raw_env._max_episode_time_sec = 60
    rows = []

    for step in range(MAX_STEPS):
        pos_xy = raw_env.data.qpos[:2].copy()
        dist, closest_idx = closest_point_on_trajectory(pos_xy, trajectory_xy)
        dist_to_end = np.linalg.norm(pos_xy - trajectory_xy[-1])
        
        # stops only if midpoint has passed and it's close to the end
        if step > 150 and dist_to_end < 0.20: # at least 3 seconds in the simulation. end tolerance: 20 cm
            print(f"     [!] Completed at step {step}")
            break

        w, x, y, z = raw_env.data.qpos[3:7]
        _, _, yaw = raw_env.euler_from_quaternion(w, x, y, z)
        wz_cmd = heading_command(pos_xy, yaw, trajectory_xy, closest_idx)

        raw_env._desired_velocity = np.array([FORWARD_SPEED, 0.0, wz_cmd])

        fresh_obs = raw_env._get_obs()
        fresh_obs_expanded = np.expand_dims(fresh_obs, axis=0)
        
        if isinstance(vec_env, VecNormalize):
            obs = vec_env.normalize_obs(fresh_obs_expanded)
        else:
            obs = fresh_obs_expanded

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        joint_angles = raw_env.data.qpos[7:19].copy()

        row_data = {
            "algorithm": algo_name,
            "trajectory": traj_name,
            "step": step,
            "time_s": step * raw_env.dt,
            "x": pos_xy[0],
            "y": pos_xy[1],
            "cross_track_error": dist,
            "wz_command": wz_cmd,
        }
        
        for j in range(12):
            row_data[f"joint_{j}"] = joint_angles[j]
            
        rows.append(row_data)

        if viewer is not None:
            draw_trajectory(viewer, trajectory_xy)
            viewer.sync()
            time.sleep(raw_env.dt)

        if done[0]:
            break

    return rows

def make_eval_env(algo_name):
    env = A1Env(render_mode=None)
    # if algo_name == "SAC":
    #     env.action_space = spaces.Box(
    #         low=-1.0,
    #         high=1.0,
    #         shape=(env.model.nu,),
    #         dtype=np.float32,
    #     )
    return env

# plotting function
def plot_joint_angles(all_rows, algo_target="PPO", traj_target="line"):
    """
    Genera un plot 2x2 elegante separando le gambe (FR, FL, RR, RL)
    """
    data = [r for r in all_rows if r["algorithm"] == algo_target and r["trajectory"] == traj_target]
    if not data:
        return
        
    time_s = [r["time_s"] for r in data]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Joint Angles during Locomotion - {algo_target} on {traj_target.capitalize()} Trajectory", fontsize=16)
    
    legs = [
        ("FR (Front Right)", 0, axs[0, 0]),
        ("FL (Front Left)", 3, axs[0, 1]),
        ("RR (Rear Right)", 6, axs[1, 0]),
        ("RL (Rear Left)", 9, axs[1, 1])
    ]
    
    joint_names = ["Abduction", "Hip", "Knee"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for leg_name, start_idx, ax in legs:
        for i, (j_name, col) in enumerate(zip(joint_names, colors)):
            joint_data = [r[f"joint_{start_idx + i}"] for r in data]
            ax.plot(time_s, joint_data, label=j_name, color=col)
            
        ax.set_title(leg_name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
    plt.tight_layout()
    filename = f"joint_angles_{algo_target}_{traj_target}.png"
    plt.savefig(filename, dpi=300)
    print(f"\n✅ Plot saved: {filename}")



def main():
    all_rows = []

    for algo_name, (algo_class, model_path, vecnorm_path) in MODELS.items():
        print(f"\nALGORITHM: {algo_name}")

        vec_env = DummyVecEnv([lambda algo=algo_name: make_eval_env(algo)])
        
        if vecnorm_path is not None:
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        model = algo_class.load(model_path, env=vec_env)
        raw_env = vec_env.envs[0] if isinstance(vec_env, VecNormalize) else vec_env.envs[0]

        viewer = None
        if RENDER:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(raw_env.model, raw_env.data)
            trunk_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = trunk_id
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -30.0
            viewer.cam.azimuth = 45.0

        for traj_name, trajectory_xy in TRAJECTORIES.items():
            print(f"  -> trajectory {traj_name}")
            rows = run_trajectory_episode(
                algo_name, model, vec_env, raw_env, traj_name, trajectory_xy, viewer=viewer
            )
            all_rows.extend(rows)

            mean_err = np.mean([r["cross_track_error"] for r in rows])
            max_err = np.max([r["cross_track_error"] for r in rows])
            print(f"     mean error {mean_err:.3f} m | max error {max_err:.3f} m | completed steps {len(rows)}")

        if viewer is not None:
            viewer.close()

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nresults saved in {OUTPUT_CSV} ({len(all_rows)} rows)")
        
        for algo in MODELS.keys():
            for traj in TRAJECTORIES.keys():
                plot_joint_angles(all_rows, algo_target=algo, traj_target=traj)

if __name__ == "__main__":
    main()