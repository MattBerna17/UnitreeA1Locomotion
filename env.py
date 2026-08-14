"""
Environment Gymnasium per la locomozione dell'Unitree A1, costruito sul tuo
a1.xml / scene.xml.

Dettagli del modello confermati leggendo l'XML (importanti per capire le
scelte fatte sotto):
- Attuatori: <position> con kp=100, forcerange=[-33.5, 33.5] Nm -> sono
  controllori PD di posizione, NON torque diretti. La policy comanda una
  posizione target per ogni giunto, MuJoCo calcola internamente la coppia.
- Ordine giunti/attuatori: FR (hip,thigh,calf), FL (hip,thigh,calf),
  RR (hip,thigh,calf), RL (hip,thigh,calf) -> 12 valori in quest'ordine
  sia in qpos[7:]/qvel[6:] che in ctrl.
- Ordine body nell'albero (per indicizzare cfrc_ext/cvel per corpo):
  0=world, 1=trunk, 2=FR_hip, 3=FR_thigh, 4=FR_calf, 5=FL_hip, 6=FL_thigh,
  7=FL_calf, 8=RR_hip, 9=RR_thigh, 10=RR_calf, 11=RL_hip, 12=RL_thigh,
  13=RL_calf -> i "piedi" (calf, dove c'e' il geom foot) sono gli indici
  [4, 7, 10, 13].
- Range giunti (dalle classi <default>): abduction +-0.803 rad,
  hip [-1.047, 4.189] rad, knee [-2.697, -0.916] rad.
- Keyframe "home": qpos/ctrl di partenza con le zampe in posizione
  standing, z=0.27.

Struttura del reward: reward di tracking della velocita' desiderata
(dominante) + bonus di gait (trotto diagonale, feet air time, healthy)
- costi di regolarizzazione (torque, action rate, joint limits, slip,
ecc). I pesi sono un punto di partenza: vanno tarati guardando
TensorBoard (usa il reward_info dettagliato qui sotto) e i video.
"""

from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco
import numpy as np
from pathlib import Path


class A1Env(MujocoEnv):
    """Environment Gymnasium per la locomozione dell'Unitree A1 (position/PD control)."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    # Indici dei body "calf" (piedi) nell'albero cinematico, vedi docstring sopra.
    FEET_BODY_INDICES = [4, 7, 10, 13]  # FR, FL, RR, RL
    # Indici dei body di hip/thigh (per rilevare collisioni indesiderate,
    # es. una gamba che urta il corpo o un'altra gamba).
    COLLISION_BODY_INDICES = [2, 3, 5, 6, 8, 9, 11, 12]

    def __init__(self, xml_path="./unitree_a1/scene.xml", **kwargs):
        model_path = Path(xml_path)
        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,  # dt(=0.002) * 10 = 0.02s -> azione a 50Hz
            observation_space=None,  # settato manualmente sotto, dopo aver definito _get_obs
            **kwargs,
        )

        self._last_render_time = -1.0
        self._max_episode_time_sec = 30.0
        self._step_count = 0

        # -----------------------------------------------------------
        # PESI reward/costo: punto di partenza, da tarare con TensorBoard.
        # Il tracking della velocita' deve restare il termine dominante.
        # -----------------------------------------------------------
        self.reward_weights = {
            "linear_vel_tracking": 4.0,
            "angular_vel_tracking": 0.1,
            "healthy": 0.5,
            "trot": 0.5,
            "feet_airtime": 0.2,
        }
        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 0.05,
            "xy_angular_vel": 0.05,
            "action_rate": 0.01,
            "joint_limit": 0.05,
            "joint_velocity": 0.01,
            "joint_acceleration": 2.5e-7,
            "orientation": 0.01,
            "collision": 0.01,
            "body_height": 0.01,
            "flight": 0.01,
            "foot_slip": 0.05,
            "roll_pitch": 0.5
        }

        self._default_joint_position = np.array(self.model.key_ctrl[0])  # dal keyframe "home"

        # Velocita' target: vx, vy, wz (m/s, m/s, rad/s)
        self._desired_velocity_min = np.array([0.5, -0.0, -0.0])
        self._desired_velocity_max = np.array([0.5, 0.0, 0.0])
        self._desired_velocity = self._sample_desired_vel()

        self._obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        self._tracking_velocity_sigma = 0.25

        # Criteri di salute per terminare l'episodio in anticipo.
        # Se il training fatica a superare pochi secondi, allargali
        # temporaneamente (curriculum), poi restringili gradualmente.
        self._healthy_z_range = (0.10, 0.80)
        self._healthy_pitch_range = (-np.deg2rad(25), np.deg2rad(25))
        self._healthy_roll_range = (-np.deg2rad(25), np.deg2rad(25))

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4, dtype=bool)

        # Range "morbido" dei giunti: penalizza solo vicino ai limiti
        # meccanici reali definiti nell'XML (90% del range libero).
        soft_margin_ratio = 0.9
        ctrl_range = self.model.actuator_ctrlrange
        offset = 0.5 * (1 - soft_margin_ratio) * (ctrl_range[:, 1] - ctrl_range[:, 0])
        self._soft_joint_range = np.copy(ctrl_range)
        self._soft_joint_range[:, 0] += offset
        self._soft_joint_range[:, 1] -= offset

        self._reset_noise_scale = 0.1
        self._last_action = np.zeros(self.model.nu)
        self._death_cause = ""

        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

    # =====================================================================
    # STEP / RESET
    # =====================================================================

    def step(self, action):
        self._step_count += 1
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._calc_reward(action)
        terminated = not self.is_healthy
        truncated = self._step_count >= (self._max_episode_time_sec / self.dt)

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        if self.render_mode == "human" and (
            self.data.time - self._last_render_time
        ) > (1.0 / self.metadata["render_fps"]):
            self.render()
            self._last_render_time = self.data.time

        self._last_action = action
        return observation, reward, terminated, truncated, info

    def reset_model(self):
        self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        self.data.ctrl[:] = self.model.key_ctrl[0] + self._reset_noise_scale * (
            self.np_random.standard_normal(self.model.nu)
        )

        self.data.qpos[:7] = [0.0, 0.0, 0.32, 1.0, 0.0, 0.0, 0.0]  # posizione + orientazione tronco
        self.data.qpos[7:] = np.array([0.0, 0.9, -1.8] * 4)         # 12 angoli zampe (stessa posa x4 zampe)
        self.data.qvel[:] = 0.0

        self.set_state(self.data.qpos, self.data.qvel)

        self._desired_velocity = self._sample_desired_vel()
        self._step_count = 0
        self._last_action = np.zeros(self.model.nu)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4, dtype=bool)
        self._last_render_time = -1.0

        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def _sample_desired_vel(self):
        return np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )

    # =====================================================================
    # HEALTH / STATO
    # =====================================================================

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        z_ok = min_z <= state[2] <= max_z

        w, x, y, z = self.data.qpos[3:7]
        roll, pitch, _ = self.euler_from_quaternion(w, x, y, z)
        min_roll, max_roll = self._healthy_roll_range
        roll_ok = min_roll <= roll <= max_roll
        min_pitch, max_pitch = self._healthy_pitch_range
        pitch_ok = min_pitch <= pitch <= max_pitch

        healthy = bool(np.isfinite(state).all() and z_ok and roll_ok and pitch_ok)
        if not healthy:
            self._death_cause = (
                f"z_ok={z_ok}(z={state[2]:.2f}) "
                f"roll_ok={roll_ok}(roll={np.rad2deg(roll):.1f}°) "
                f"pitch_ok={pitch_ok}(pitch={np.rad2deg(pitch):.1f}°)"
            )
            # print(self._death_cause)
        return healthy

    @property
    def proj_gravity(self):
        """Gravita' nel frame del corpo: dice quanto/dove e' inclinato il tronco."""
        trunk_id = self.model.body("trunk").id
        R_body_to_world = self.data.xmat[trunk_id].reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -1.0])
        return R_body_to_world.T @ gravity_world

    @property
    def feet_contact_forces(self):
        forces = self.data.cfrc_ext[self.FEET_BODY_INDICES]
        return np.linalg.norm(forces, axis=1)

    # =====================================================================
    # REWARD (termini positivi)
    # =====================================================================

    @property
    def linear_velocity_tracking_reward(self):
        vel_sqr_error = np.sum(
            np.square(self._desired_velocity[:2] - self.data.qvel[:2])
        )
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def angular_velocity_tracking_reward(self):
        vel_sqr_error = np.square(self._desired_velocity[2] - self.data.qvel[5])
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def healthy_reward(self):
        return float(self.is_healthy)

    @property
    def feet_air_time_reward(self):
        """Premia un tempo in aria vicino al target (gait naturale, non strisciato)."""
        curr_contact = self.feet_contact_forces > 1.0
        first_contact = curr_contact & ~self._last_contacts

        self._feet_air_time[~curr_contact] += self.dt
        air_time = self._feet_air_time.copy()

        target = 0.25  # tempo in aria atteso per v desiderata = 0.5 m/s
        sigma = 0.10
        reward = np.sum(
            np.exp(-np.square(air_time - target) / sigma**2) * first_contact
        )

        self._feet_air_time[curr_contact] = 0.0
        self._last_contacts = curr_contact
        return reward

    @property
    def trot_reward(self):
        """
        Premia il pattern diagonale del trotto: FR+RL si muovono insieme,
        cosi' come FL+RR. Ordine feet_contact_forces: [FR, FL, RR, RL].
        """
        curr_contact = self.feet_contact_forces > 1.0
        diag1_sync = float(curr_contact[0] == curr_contact[3])  # FR & RL
        diag2_sync = float(curr_contact[1] == curr_contact[2])  # FL & RR
        return 0.5 * (diag1_sync + diag2_sync)

    # =====================================================================
    # COSTI (termini negativi, regolarizzazione)
    # =====================================================================

    @property
    def flight_cost(self):
        """Penalizza il non avere nessun piede a terra (volo/salto inutile)."""
        return float(not np.any(self.feet_contact_forces > 1.0))

    @property
    def body_height_cost(self):
        target_height = 0.27  # altezza dal keyframe "home"
        return np.square(self.data.qpos[2] - target_height)

    @property
    def non_flat_base_cost(self):
        return np.sum(np.square(self.proj_gravity[:2]))

    @property
    def collision_cost(self):
        return np.sum(
            1.0
            * (np.linalg.norm(self.data.cfrc_ext[self.COLLISION_BODY_INDICES]) > 0.1)
        )

    @property
    def joint_limit_cost(self):
        pos = self.data.qpos[7:]
        out_of_range = (self._soft_joint_range[:, 0] - pos).clip(min=0.0) + (
            pos - self._soft_joint_range[:, 1]
        ).clip(min=0.0)
        return np.sum(out_of_range)

    @property
    def torque_cost(self):
        return np.sum(np.square(self.data.qfrc_actuator[-self.model.nu:]))

    @property
    def vertical_velocity_cost(self):
        return np.square(self.data.qvel[2])

    @property
    def xy_angular_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[3:5]))

    def action_rate_cost(self, action):
        return np.sum(np.square(self._last_action - action))

    @property
    def joint_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[6:]))

    @property
    def acceleration_cost(self):
        return np.sum(np.square(self.data.qacc[6:]))

    @property
    def foot_slip_cost(self):
        """
        Penalizza lo scivolamento: quando un piede tocca terra, la sua
        velocita' orizzontale dovrebbe essere ~0. data.cvel per body ha
        formato [ang(3), lin(3)] -> componenti lineari xy = indici [3:5].
        """
        curr_contact = self.feet_contact_forces > 1.0
        feet_xy_vel = self.data.cvel[self.FEET_BODY_INDICES][:, 3:5]
        return np.sum(np.square(feet_xy_vel) * curr_contact[:, None])
    
    @property
    def roll_pitch_cost(self):
        """
        Penalizza roll e pitch eccessivi, per scoraggiare la policy dal
        ribaltarsi lateralmente/longitudinalmente. Quadratico: piccole
        inclinazioni costano poco, inclinazioni forti costano molto di più.
        """
        w, x, y, z = self.data.qpos[3:7]
        roll, _, _ = self.euler_from_quaternion(w, x, y, z)
        return np.square(roll)

    # =====================================================================
    # CALCOLO REWARD TOTALE
    # =====================================================================

    def _calc_reward(self, action):
        linear_vel_tracking_reward = (
            self.linear_velocity_tracking_reward
            * self.reward_weights["linear_vel_tracking"]
        )
        angular_vel_tracking_reward = (
            self.angular_velocity_tracking_reward
            * self.reward_weights["angular_vel_tracking"]
        )
        healthy_reward = self.healthy_reward * self.reward_weights["healthy"]
        feet_air_time_reward = (
            self.feet_air_time_reward * self.reward_weights["feet_airtime"]
        )
        trot_reward = self.trot_reward * self.reward_weights["trot"]

        rewards = (
            linear_vel_tracking_reward
            + angular_vel_tracking_reward
            + healthy_reward
            + feet_air_time_reward
            + trot_reward
        )

        body_height_cost = self.body_height_cost * self.cost_weights["body_height"]
        ctrl_cost = self.torque_cost * self.cost_weights["torque"]
        action_rate_cost = (
            self.action_rate_cost(action) * self.cost_weights["action_rate"]
        )
        vertical_vel_cost = (
            self.vertical_velocity_cost * self.cost_weights["vertical_vel"]
        )
        xy_angular_vel_cost = (
            self.xy_angular_velocity_cost * self.cost_weights["xy_angular_vel"]
        )
        joint_limit_cost = self.joint_limit_cost * self.cost_weights["joint_limit"]
        joint_velocity_cost = (
            self.joint_velocity_cost * self.cost_weights["joint_velocity"]
        )
        joint_acceleration_cost = (
            self.acceleration_cost * self.cost_weights["joint_acceleration"]
        )
        orientation_cost = self.non_flat_base_cost * self.cost_weights["orientation"]
        collision_cost = self.collision_cost * self.cost_weights["collision"]
        flight_cost = self.flight_cost * self.cost_weights["flight"]
        foot_slip_cost = self.foot_slip_cost * self.cost_weights["foot_slip"]
        roll_pitch_cost = self.roll_pitch_cost * self.cost_weights["roll_pitch"]

        costs = (
            ctrl_cost
            + body_height_cost
            + action_rate_cost
            + vertical_vel_cost
            + xy_angular_vel_cost
            + joint_limit_cost
            + joint_velocity_cost
            + joint_acceleration_cost
            + orientation_cost
            + collision_cost
            + flight_cost
            + foot_slip_cost
            + roll_pitch_cost
        )

        reward = rewards - costs

        # Log granulare per TensorBoard: capire quale termine domina e'
        # essenziale per tarare i pesi in modo consapevole, non a caso.
        reward_info = {
            "reward_linear_vel_tracking": linear_vel_tracking_reward,
            "reward_angular_vel_tracking": angular_vel_tracking_reward,
            "reward_healthy": healthy_reward,
            "reward_feet_air_time": feet_air_time_reward,
            "reward_trot": trot_reward,
            "reward_total_positive": rewards,
            "cost_torque": -ctrl_cost,
            "cost_body_height": -body_height_cost,
            "cost_action_rate": -action_rate_cost,
            "cost_vertical_vel": -vertical_vel_cost,
            "cost_xy_angular_vel": -xy_angular_vel_cost,
            "cost_joint_limit": -joint_limit_cost,
            "cost_joint_velocity": -joint_velocity_cost,
            "cost_joint_acceleration": -joint_acceleration_cost,
            "cost_orientation": -orientation_cost,
            "cost_collision": -collision_cost,
            "cost_flight": -flight_cost,
            "cost_foot_slip": -foot_slip_cost,
            "cost_total": -costs,
        }

        return reward, reward_info

    # =====================================================================
    # OSSERVAZIONE
    # =====================================================================

    def _get_obs(self):
        dofs_position = self.data.qpos[7:] - self._default_joint_position

        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3]
        base_angular_velocity = velocity[3:6]
        dofs_velocity = velocity[6:]

        curr_obs = np.concatenate(
            (
                base_linear_velocity * self._obs_scale["linear_velocity"],
                base_angular_velocity * self._obs_scale["angular_velocity"],
                self.proj_gravity,
                self._desired_velocity * self._obs_scale["linear_velocity"],
                dofs_position * self._obs_scale["dofs_position"],
                dofs_velocity * self._obs_scale["dofs_velocity"],
                self._last_action,
            )
        ).clip(-self._clip_obs_threshold, self._clip_obs_threshold)

        return curr_obs

    # =====================================================================
    # UTILS
    # =====================================================================

    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        """Converte un quaternione in angoli di Eulero (roll, pitch, yaw), in radianti."""
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z