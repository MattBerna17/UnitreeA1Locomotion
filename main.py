import mujoco
from stable_baselines3 import PPO
from env1 import A1Env

env = A1Env(render_mode="human")

model = PPO.load("ppo_a1.zip", env=env)

obs, info = env.reset()
action = env._default_joint_position

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    # print([key for key in info.keys()])
    # print(info.get("rewards"))
    # print(info.get("costs"))

    if terminated or truncated:
        obs, info = env.reset()

env.close()