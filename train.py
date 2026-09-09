import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_checker import check_env
from env1 import A1Env

ALGORITHM = "PPO" 
TOTAL_TIMESTEPS = 1_000_000

N_ENVS = 4
if ALGORITHM == "PPO":
    ALGO_KWARGS = dict(n_steps=2048, batch_size=64)
    MODEL_NAME = "ppo_a1_trajectories"
    VECNORM_NAME = "ppo_vecnormalize_a1_trajectories"
elif ALGORITHM == "SAC":
    MODEL_NAME = "sac_a1_trajectories"
else:
    raise ValueError(f"Algorithm not supported: {ALGORITHM}")

def make_env():
    env = A1Env(render_mode=None)
    # if ALGORITHM == "SAC" or ALGORITHM == "PPO":
    #     env.action_space = spaces.Box(
    #         low=-1.0,
    #         high=1.0,
    #         shape=(env.model.nu,),
    #         dtype=np.float32,
    #     )
    return env

def main():
    check_env(make_env(), warn=True)
    print("check_env passed")

    SEEDS = [15, 42, 99]
    for seed in SEEDS:
        vec_env = make_vec_env(make_env, n_envs=N_ENVS, seed=seed)

        if ALGORITHM == "PPO":
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                seed=seed,
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
                ent_coef="auto",
                seed=seed
            )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            tb_log_name=f"{ALGORITHM.lower()}_seed_{seed}",
            progress_bar=True,
        )

        model.save(f"{MODEL_NAME}_seed_{seed}")
        
        if ALGORITHM == "PPO":
            vec_env.save(f"{VECNORM_NAME}_seed_{seed}.pkl")
            print(f"Saved: {MODEL_NAME}.zip and {VECNORM_NAME}_seed_{seed}")
        else:
            print(f"Saved: {MODEL_NAME}.zip")

if __name__ == "__main__":
    main()