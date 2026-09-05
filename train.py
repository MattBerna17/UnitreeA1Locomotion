import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_checker import check_env
from env1 import A1Env

ALGORITHM = "SAC" 
TOTAL_TIMESTEPS = 3_000_000

if ALGORITHM == "PPO":
    N_ENVS = 8
    ALGO_KWARGS = dict(n_steps=2048, batch_size=64)
    MODEL_NAME = "ppo_a1_trajectories"
    VECNORM_NAME = "ppo_vecnormalize_a1_trajectories.pkl"
elif ALGORITHM == "SAC":
    N_ENVS = 4
    TOTAL_TIMESTEPS = 1_000_000
    MODEL_NAME = "sac_a1_trajectories"
else:
    raise ValueError(f"Algorithm not supported: {ALGORITHM}")

def make_env():
    env = A1Env(render_mode=None)
    if ALGORITHM == "SAC":
        env.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env.model.nu,),
            dtype=np.float32,
        )
    return env

def main():
    check_env(make_env(), warn=True)
    print("check_env passed")

    vec_env = make_vec_env(make_env, n_envs=N_ENVS)

    if ALGORITHM == "PPO":
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            **ALGO_KWARGS,
        )
    else:
        # SAC instanziato direttamente sul DummyVecEnv senza normalizzazione
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=8,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto"
        )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=ALGORITHM.lower(),
        progress_bar=True,
    )

    model.save(MODEL_NAME)
    
    if ALGORITHM == "PPO":
        vec_env.save(VECNORM_NAME)
        print(f"Saved: {MODEL_NAME}.zip and {VECNORM_NAME}")
    else:
        print(f"Saved: {MODEL_NAME}.zip")

if __name__ == "__main__":
    main()