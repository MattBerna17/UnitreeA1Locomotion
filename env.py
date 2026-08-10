import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import mujoco
import mujoco.viewer
import time

class Environment(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # 1. Carichiamo la fisica direttamente, come facevi nel tuo script base
        self.model = mujoco.MjModel.from_xml_path("unitree_a1/scene.xml")
        self.data = mujoco.MjData(self.model)
        
        # 2. Definiamo gli spazi Gym
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)
        
        # 3. Inizializziamo il visualizzatore nativo di MuJoCo
        self.viewer = None
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Resettiamo la fisica
        mujoco.mj_resetData(self.model, self.data)
        
        # Posizioniamo il robot a 32 cm da terra in piedi
        self.data.qpos[:7] = [0.0, 0.0, 0.32, 1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:] = np.array([0.0, 0.9, -1.8] * 4)
        mujoco.mj_forward(self.model, self.data)
        
        # Aggiorniamo la grafica al reset
        if self.viewer is not None:
            self.viewer.sync()
        
        # Ritorna l'osservazione dummy
        obs = np.zeros(35, dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Applichiamo l'azione generata ai motori
        self.data.ctrl[:] = action
        
        # Facciamo avanzare la simulazione fisica (il cuore pulsante)
        mujoco.mj_step(self.model, self.data)
        
        # Se la grafica è attiva, sincronizziamo i frame e limitiamo gli FPS
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()
            time.sleep(self.model.opt.timestep) 
        
        # Dati fittizi in attesa di calcolare Reward e Termination veri
        obs = np.zeros(35, dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        # Non serve codice qui: launch_passive si aggiorna da solo nello step()
        pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()