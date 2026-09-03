import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env1 import A1Env

# --- Caricamento modello + VecNormalize (stessa logica del tuo main.py) ---
vec_env = DummyVecEnv([lambda: A1Env(render_mode=None)])
vec_env = VecNormalize.load("vecnormalize_a1_thr.pkl", vec_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load("ppo_a1_thr.zip", env=vec_env)

# --- Accesso all'env "vero" sotto i wrapper, per leggere qvel/qacc grezzi ---
raw_env = vec_env.envs[0]  # A1Env non wrappato, accesso diretto ai dati MuJoCo

obs = vec_env.reset()

vel_samples, acc_samples, action_diff_samples = [], [], []
n_steps = 5000

for _ in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    prev_action = raw_env._last_action.copy()  # prima dello step, per calcolare il diff corretto

    obs, reward, done, info = vec_env.step(action)

    vel_samples.append(np.abs(raw_env.data.qvel[6:]))
    acc_samples.append(np.abs(raw_env.data.qacc[6:]))
    action_diff_samples.append(np.abs(action[0] - prev_action))  # action[0]: DummyVecEnv aggiunge una dim batch

    if done[0]:
        obs = vec_env.reset()

vel_samples = np.concatenate(vel_samples)
acc_samples = np.concatenate(acc_samples)
action_diff_samples = np.concatenate(action_diff_samples)

for name, samples in [("joint_velocity", vel_samples), ("acceleration", acc_samples), ("action_diff", action_diff_samples)]:
    print(f"{name}: p50={np.percentile(samples,50):.3f}  p75={np.percentile(samples,75):.3f}  "
          f"p90={np.percentile(samples,90):.3f}  p95={np.percentile(samples,95):.3f}  max={samples.max():.3f}")