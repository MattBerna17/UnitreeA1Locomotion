"""
Valuta uno o piu' modelli RL (es. PPO, SAC) sul tracking di 3 traiettorie
di riferimento (linea, sinusoide, cerchio), salvando in CSV la distanza
(cross-track error) tra il robot e la traiettoria ad ogni step.

IMPORTANTE - limite del tuo A1Env attuale:
il tuo environment e' stato allenato con un comando di velocita' fisso
[vx, 0, 0] (nessuna sterzata). Qui, per seguire traiettorie curve, viene
calcolato un comando di velocita' angolare (wz) con un semplice
controllore proporzionale sull'errore di heading verso un punto della
traiettoria un po' avanti (lookahead), e iniettato in
env._desired_velocity ad ogni step.

La policy PUO' generalizzare a piccoli wz anche se allenata solo con
wz=0 (fa parte dell'osservazione), ma non e' garantito, specialmente
sul cerchio (curvatura piu' stretta). Se il tracking sulla sinusoide/
cerchio e' pessimo mentre sulla linea e' buono, il problema e' quasi
certamente questo: servirebbe un fine-tuning con un range di wz
campionato durante il training (stesso principio del curriculum sulla
velocita' lineare che avevi gia' fatto), non un bug in questo script.

Uso:
    python evaluate_trajectories.py
"""

import csv
import numpy as np
import mujoco

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env1 import A1Env
from trajectories import TRAJECTORIES, closest_point_on_trajectory


# =========================================================================
# CONFIGURAZIONE
# =========================================================================

# Aggiungi qui i modelli da confrontare. Ogni voce: (classe SB3, path .zip, path vecnormalize.pkl)
MODELS = {
    "PPO": (PPO, "ppo_a1_trajectories.zip", "ppo_vecnormalize_a1_trajectories.pkl"),
    # "SAC": (SAC, "sac_a1_trajectories.zip", "sac_vecnormalize_a1_trajectories.pkl"),
}

RENDER = True          # True per vedere robot + traiettoria in verde nel viewer MuJoCo
MAX_STEPS = 3000         # ~60s a 50Hz di controllo
FORWARD_SPEED = 0.3     # deve combaciare col target usato in training

# Controllore di sterzata (proporzionale sull'errore di heading)
LOOKAHEAD_POINTS = 8    # quanti punti avanti sulla traiettoria guardare
KP_HEADING = 2.0
WZ_MAX = 0.2            # satura il comando angolare per restare in un range "ragionevole"

OUTPUT_CSV = "trajectory_tracking_results.csv"
TRAJECTORY_Z = 0.02      # altezza da terra della linea verde disegnata


# =========================================================================
# VISUALIZZAZIONE: disegna la traiettoria come linea verde nel viewer
# =========================================================================

def draw_trajectory(viewer, trajectory_xy, z=TRAJECTORY_Z):
    """
    Disegna la traiettoria come sequenza di capsule verdi nella "user
    scene" del viewer passivo di MuJoCo (geometria solo visiva, non
    interagisce con la fisica).
    """
    scn = viewer.user_scn
    scn.ngeom = 0
    step = max(1, len(trajectory_xy) // 300)  # sottocampiona per non sovraccaricare il rendering
    for i in range(0, len(trajectory_xy) - step, step):
        p1 = np.array([trajectory_xy[i, 0], trajectory_xy[i, 1], z])
        p2 = np.array([trajectory_xy[i + step, 0], trajectory_xy[i + step, 1], z])
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008, p1, p2
        )
        geom.rgba[:] = [0.0, 1.0, 0.0, 1.0]
        scn.ngeom += 1


# =========================================================================
# CONTROLLO DI STERZATA VERSO LA TRAIETTORIA
# =========================================================================

def heading_command(position_xy, yaw, trajectory_xy, closest_idx):
    """
    Calcola un comando di velocita' angolare (wz) proporzionale
    all'errore tra lo heading attuale del robot e la direzione verso
    un punto un po' avanti sulla traiettoria (lookahead).
    """
    target_idx = min(closest_idx + LOOKAHEAD_POINTS, len(trajectory_xy) - 1)
    target = trajectory_xy[target_idx]

    desired_yaw = np.arctan2(target[1] - position_xy[1], target[0] - position_xy[0])
    heading_error = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))

    wz = np.clip(KP_HEADING * heading_error, -WZ_MAX, WZ_MAX)
    return wz


# =========================================================================
# ESECUZIONE DI UN EPISODIO DI TRACKING
# =========================================================================

def run_trajectory_episode(algo_name, model, vec_env, raw_env, traj_name, trajectory_xy, viewer=None):
    obs = vec_env.reset()

    # Riposiziona il robot all'inizio della traiettoria (tutte partono
    # tangenti a +x nell'origine, vedi trajectories.py, quindi non serve
    # ruotare l'orientazione iniziale).
    raw_env.data.qpos[0:2] = trajectory_xy[0]
    mujoco.mj_forward(raw_env.model, raw_env.data)

    rows = []
    for step in range(MAX_STEPS):
        pos_xy = raw_env.data.qpos[:2].copy()
        dist, closest_idx = closest_point_on_trajectory(pos_xy, trajectory_xy)

        w, x, y, z = raw_env.data.qpos[3:7]
        _, _, yaw = raw_env.euler_from_quaternion(w, x, y, z)
        wz_cmd = heading_command(pos_xy, yaw, trajectory_xy, closest_idx)

        # Inietta il comando di velocita' calcolato
        raw_env._desired_velocity = np.array([FORWARD_SPEED, 0.0, wz_cmd])

        # FIX: Rigenera l'osservazione e normalizzala per SB3 (richiede la dimensione batch)
        fresh_obs = raw_env._get_obs()
        obs = vec_env.normalize_obs(np.expand_dims(fresh_obs, axis=0))

        # Ora il modello vede la correzione in tempo reale
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # print(f"[STEP {step}] ACTION: {action}")

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

        if done[0]:
            break

    return rows


# =========================================================================
# MAIN: cicla su tutti i modelli e tutte le traiettorie
# =========================================================================

def main():
    all_rows = []

    for algo_name, (algo_class, model_path, vecnorm_path) in MODELS.items():
        print(f"\n=== Modello: {algo_name} ===")


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

        for traj_name, trajectory_xy in TRAJECTORIES.items():
            print(f"  -> traiettoria: {traj_name}")
            rows = run_trajectory_episode(
                algo_name, model, vec_env, raw_env, traj_name, trajectory_xy, viewer=viewer
            )
            all_rows.extend(rows)

            mean_err = np.mean([r["cross_track_error"] for r in rows])
            max_err = np.max([r["cross_track_error"] for r in rows])
            print(f"     errore medio: {mean_err:.3f} m | errore massimo: {max_err:.3f} m | step completati: {len(rows)}")

        if viewer is not None:
            viewer.close()

    # --- Salvataggio CSV ---
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nRisultati salvati in: {OUTPUT_CSV} ({len(all_rows)} righe)")


if __name__ == "__main__":
    main()
