import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env1 import A1Env
from stable_baselines3.common.env_checker import check_env


env = A1Env(xml_path="./unitree_a1/scene.xml", render_mode=None)
check_env(env, warn=True)
print("✅ check_env passed")

vec_env = make_vec_env(lambda: A1Env(render_mode=None), n_envs=8)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    n_steps=2048,
    batch_size=64,
)
model.learn(total_timesteps=10_000_000)

model.save("ppo_a1")
vec_env.save("vecnormalize_a1.pkl")