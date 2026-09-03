"""fine_tuning_steering.py"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env1 import A1Env

N_ENVS = 8
OLD_MODEL_PATH = "ppo_a1.zip"
OLD_VECNORM_PATH = "vecnormalize_a1.pkl"
NEW_MODEL_PATH = "ppo_a1_yaw_v1"
NEW_VECNORM_PATH = "vecnormalize_a1_yaw_v1.pkl"

# vx resta quello che già funziona, si allarga SOLO wz (indice 2)
NEW_VEL_MIN = np.array([0.3, 0.0, -0.5])   # piccolo primo gradino: ±0.2 rad/s
NEW_VEL_MAX = np.array([0.3, 0.0,  0.5])

FINE_TUNE_STEPS = 1_000_000

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardComponentsCallback(BaseCallback):
    """Logga su TensorBoard le singole componenti di reward_info."""

    def _on_step(self) -> bool:
        infos = self.locals["infos"]  # lista di info, una per ogni env parallelo
        # media tra gli n_envs paralleli, per ogni chiave presente
        keys = [k for k in infos[0].keys() if k.startswith("reward_") or k.startswith("cost_")]
        for key in keys:
            values = [info[key] for info in infos if key in info]
            if values:
                self.logger.record(f"reward_components/{key}", np.mean(values))
        return True


def make_env():
    return A1Env(render_mode=None)

def main():
    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
    vec_env = VecNormalize.load(OLD_VECNORM_PATH, vec_env)
    vec_env.training = True
    vec_env.norm_reward = True

    vec_env.set_attr("_desired_velocity_min", NEW_VEL_MIN)
    vec_env.set_attr("_desired_velocity_max", NEW_VEL_MAX)

    model = PPO.load(OLD_MODEL_PATH, env=vec_env)
    model.learn(total_timesteps=FINE_TUNE_STEPS, reset_num_timesteps=False, tb_log_name="yaw_0.2", callback=RewardComponentsCallback(), progress_bar=True)

    model.save(NEW_MODEL_PATH)
    vec_env.save(NEW_VECNORM_PATH)

if __name__ == "__main__":
    main()