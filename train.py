# training PPO, A2C or SAC on the A1 robot

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_checker import check_env
from env1 import A1Env

ALGORITHM = "SAC" # change to "A2C" or "SAC" to train another algorithm
TOTAL_TIMESTEPS = 3_000_000


if ALGORITHM == "PPO":
    N_ENVS = 8
    ALGO_KWARGS = dict(
        n_steps=2048,
        batch_size=64,
    )
    NORM_REWARD = True
    MODEL_NAME = "ppo_a1_trajectories"
    VECNORM_NAME = "ppo_vecnormalize_a1_trajectories.pkl"

elif ALGORITHM == "A2C":
    N_ENVS = 8
    ALGO_KWARGS = dict(
        n_steps=2048,
        ent_coef=0.01,
    )
    NORM_REWARD = True
    MODEL_NAME = "a2c_a1_trajectories"
    VECNORM_NAME = "a2c_vecnormalize_a1_trajectories.pkl"

elif ALGORITHM == "SAC":
    N_ENVS = 4
    TOTAL_TIMESTEPS = 1_000_000
    NORM_REWARD = False
    MODEL_NAME = "sac_a1_trajectories"
    VECNORM_NAME = "sac_vecnormalize_a1_trajectories.pkl"

else:
    raise ValueError(f"Algorithm not supported: {ALGORITHM}")


def make_env():
    env = A1Env(render_mode=None)

    if ALGORITHM == "SAC":
        # override the action space only for SAC
        env.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(env.model.nu,),
            dtype=np.float32,
        )

    return env


def main():
    # check the environment with the final action space
    probe_env = make_env()
    check_env(probe_env, warn=True)
    print("check_env passed")

    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=NORM_REWARD,
    )

    if ALGORITHM == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            **ALGO_KWARGS,
        )

    elif ALGORITHM == "A2C":
        model = A2C(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            **ALGO_KWARGS,
        )

    else:
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
    vec_env.save(VECNORM_NAME)

    print(f"Saved: {MODEL_NAME}.zip and {VECNORM_NAME}")


if __name__ == "__main__":
    main()
