# evaluates rl models on tracking reference trajectories
# saves cross track error to csv
# calculates and injects angular velocity command
# usage python evaluate_trajectories.py

import csv
import numpy as np
import mujoco

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env1 import A1Env
from trajectories import TRAJECTORIES, closest_point_on_trajectory

# configuration
# add models to compare here
MODELS = {
    "PPO": (PPO, "ppo_a1_trajectories.zip", "ppo_vecnormalize_a1_trajectories.pkl"),
    "SAC": (SAC, "sac_a1_trajectories.zip", "sac_vecnormalize_a1_trajectories.pkl"),
}

RENDER = True          # true to see robot and trajectory in viewer
MAX_STEPS = 3000       # max steps for 60s at 50hz
FORWARD_SPEED = 0.3    # must match target used in training

# steering controller based on heading error
LOOKAHEAD_POINTS = 8   # points to look ahead on the trajectory
KP_HEADING = 2.0
WZ_MAX = 1.0           # saturates angular command to stay in reasonable range

OUTPUT_CSV = "trajectory_tracking_results.csv"
TRAJECTORY_Z = 0.02    # height of the drawn green line

# visualization
def draw_trajectory(viewer, trajectory_xy, z=TRAJECTORY_Z):
    # draws trajectory as sequence of green capsules in passive viewer
    scn = viewer.user_scn
    scn.ngeom = 0
    step = max(1, len(trajectory_xy) // 300)  # downsample to avoid overloading rendering
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
    # calculates angular velocity command proportional to heading error
    target_idx = min(closest_idx + LOOKAHEAD_POINTS, len(trajectory_xy) - 1)
    target = trajectory_xy[target_idx]

    desired_yaw = np.arctan2(target[1] - position_xy[1], target[0] - position_xy[0])
    heading_error = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))

    wz = np.clip(KP_HEADING * heading_error, -WZ_MAX, WZ_MAX)
    return wz

# tracking episode execution
def run_trajectory_episode(algo_name, model, vec_env, raw_env, traj_name, trajectory_xy, viewer=None):
    obs = vec_env.reset()

    # resets robot at the beginning of the trajectory
    raw_env.data.qpos[0:2] = trajectory_xy[0]
    mujoco.mj_forward(raw_env.model, raw_env.data)

    rows = []
    # vel_over_time = []
    for step in range(MAX_STEPS):
        pos_xy = raw_env.data.qpos[:2].copy()

        # arrival condition
        # measures distance between robot and final point
        dist_to_end = np.linalg.norm(pos_xy - trajectory_xy[-1])
        if dist_to_end < 0.15 and traj_name != "circle":  # 15 cm tolerance
            print(f"     [!] target reached at step {step}")
            break

        dist, closest_idx = closest_point_on_trajectory(pos_xy, trajectory_xy)

        w, x, y, z = raw_env.data.qpos[3:7]
        _, _, yaw = raw_env.euler_from_quaternion(w, x, y, z)
        wz_cmd = heading_command(pos_xy, yaw, trajectory_xy, closest_idx)

        # injects calculated velocity command
        raw_env._desired_velocity = np.array([FORWARD_SPEED, 0.0, wz_cmd])

        # regenerates observation and normalizes for sb3
        fresh_obs = raw_env._get_obs()
        obs = vec_env.normalize_obs(np.expand_dims(fresh_obs, axis=0))

        # model sees correction in real time
        action, _ = model.predict(obs, deterministic=True)

        # if step < 10:
        #     print(f"step {step}: desired_vel={raw_env._desired_velocity}, action={action}")
        
        obs, reward, done, info = vec_env.step(action)

        rows.append({
            "algorithm": algo_name,
            "trajectory": traj_name,
            "step": step,
            "time_s": step * raw_env.dt,
            "x": pos_xy[0],
            "y": pos_xy[1],
            "cross_track_error": dist,
            "wz_command": wz_cmd,
        })

        if viewer is not None:
            draw_trajectory(viewer, trajectory_xy)
            viewer.sync()
        # vel_over_time.append(raw_env.data.qvel[6:].copy())

        if done[0]:
            break
    
    # vel_array = np.array(vel_over_time)
    # print("std per joint:", vel_array.std(axis=0))

    return rows

# main loop
def main():
    all_rows = []

    for algo_name, (algo_class, model_path, vecnorm_path) in MODELS.items():
        print(f"\nALGORITHM: {algo_name}")

        vec_env = DummyVecEnv([lambda: A1Env(render_mode=None)])
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        model = algo_class.load(model_path, env=vec_env)
        raw_env = vec_env.envs[0]

        viewer = None
        if RENDER:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(raw_env.model, raw_env.data)
            trunk_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = trunk_id
            viewer.cam.distance = 5
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

    # save csv
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nresults saved in {OUTPUT_CSV} ({len(all_rows)} rows)")

if __name__ == "__main__":
    main()