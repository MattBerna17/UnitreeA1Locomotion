import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env1 import A1Env


vec_env = DummyVecEnv([
    lambda: A1Env(render_mode="human")
])
vec_env = VecNormalize.load("vecnormalize_a1.pkl", vec_env)
vec_env.training = False
vec_env.norm_reward = False


# load the model
model = PPO.load("ppo_a1.zip", env=vec_env)
obs = vec_env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = vec_env.step(action)

    if done[0]:
        obs = vec_env.reset()