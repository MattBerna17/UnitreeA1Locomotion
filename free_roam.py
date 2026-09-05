import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env1 import A1Env

algorithm = "SAC"

def make_eval_env():
    env = A1Env(render_mode="human")
    if algorithm == "SAC":
        env.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env.model.nu,),
            dtype=np.float32,
        )
    return env

if algorithm == "PPO":
    vec_env = DummyVecEnv([make_eval_env])
    vec_env = VecNormalize.load("ppo_vecnormalize_a1_trajectories.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load("ppo_a1_trajectories", env=vec_env)

elif algorithm == "SAC":
    # Creazione ambiente liscio senza wrapper di normalizzazione
    vec_env = DummyVecEnv([make_eval_env])
    model = SAC.load("sac_a1_trajectories", env=vec_env)

else:
    raise ValueError("Non valid algorithm")

obs = vec_env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    if done[0]:
        obs = vec_env.reset()