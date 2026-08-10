from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco

import numpy as np
from pathlib import Path


DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 1,
    "trackbodyid": -1,
    "type": 2,
}


class A1Env(MujocoEnv):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, ctrl_type="torque", **kwargs):
        model_path = Path(f"./unitree_a1/scene.xml")
        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,  # perform an action every 10 frames (dt(=0.002) * 10 = 0.02 seconds -> 50hz action rate)
            observation_space=None,  # manually set afterwards
            # default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 60,
        }
        self._last_render_time = -1.0 # when the last rednering has happened
        self._max_episode_time_sec = 15.0 # 15s max for each episode
        self._step = 0 # number of decisions of the policy taken in an episode

        # weights for the reward and cost functions
        self.reward_weights = {
            "linear_vel_tracking": 2.0, # main goal whose weight encourages the model to move
            "angular_vel_tracking": 1.0,
            "healthy": 0.0, # currently does not contribute
            "feet_airtime": 1.0, # encourages the model to keep legs up. prevents the robot from dragging itself
        }
        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 2.0, # penalizes useless jumps and encourages stable z position
            "xy_angular_vel": 0.05, # if it rotates on x or y
            "action_rate": 0.01, # changes action too often
            "joint_limit": 10.0, # penalize exagerated joints movements
            "joint_velocity": 0.01,
            "joint_acceleration": 2.5e-7, 
            "orientation": 1.0,
            "collision": 1.0,
            "default_joint_position": 0.1
        }

        self._gravity_vector = np.array(self.model.opt.gravity) # [0, 0, -9.81] gravity on mujoco simulation environment
        self._default_joint_position = np.array(self.model.key_ctrl[0]) # position actuators (not torque actuators!) TODO: check if A1 has position or torque actuators

        # vx (m/s), vy (m/s), wz (rad/s)
        # desider velocity 0.5 m/s on the x axis, and avoid movement on the y and z axis
        self._desired_velocity_min = np.array([0.5, -0.0, -0.0])
        self._desired_velocity_max = np.array([0.5, 0.0, 0.0])
        self._desired_velocity = self._sample_desired_vel()  # [0.5, 0.0, 0.0]
        # homogeneous values in input to the NN
        self._obs_scale = {
            "linear_velocity": 2.0,
            "angular_velocity": 0.25,
            "dofs_position": 1.0,
            "dofs_velocity": 0.05,
        }
        self._tracking_velocity_sigma = 0.25 # how much the reward lowers when real velocity drifts away from target velocity

        # metrics used to determine if the episode should be terminated
        self._healthy_z_range = (0.22, 0.65) # z < 0.22 probably fallen. z > 0.65 probably jumping or exploded
        # limitations on the rotation on x and y axis to avoid falling
        self._healthy_pitch_range = (-np.deg2rad(10), np.deg2rad(10))
        self._healthy_roll_range = (-np.deg2rad(10), np.deg2rad(10))

        self._feet_air_time = np.zeros(4) # how long each foot has been in the air
        self._last_contacts = np.zeros(4) # if each foot, during the previous timestep, was in contact with something
        """
        trunk
        ├─ FR_hip
        │   └─ FR_thigh
        │       └─ FR_calf
        ├─ FL_hip
        │   └─ FL_thigh
        │       └─ FL_calf
        ├─ RR_hip
        │   └─ RR_thigh
        │       └─ RR_calf
        └─ RL_hip
            └─ RL_thigh
                └─ RL_calf
        """
        self._cfrc_ext_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL
        self._cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12] # sensors for possible collisions

        # non-penalized degrees of freedom range of the control joints
        dof_position_limit_multiplier = 0.9  # the % of the range that is not penalized -> avoid exhagerated movement of the joint
        ctrl_range_offset = (
            0.5 * (1 - dof_position_limit_multiplier) * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        ) # safe range of the joints
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0.1 # to add noise in the initial position of the robot

        self._last_action = np.zeros(12) # vector of the last values of the action

        # clip observations to [-100, +100]
        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float64
        )

        # # Feet site names to index mapping
        # # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-site
        # # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtobj
        # feet_site = [
        #     "FR",
        #     "FL",
        #     "RR",
        #     "RL",
        # ]
        # self._feet_site_name_to_id = {
        #     f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
        #     for f in feet_site
        # }

        self._main_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk") # define the trunk of the robot

    def step(self, action):
        self._step += 1
        self.do_simulation(action, self.frame_skip) # for each step, self.frame_skip frames in mujoco

        observation = self._get_obs() # after the action execution, get the resulting observations (new position of the robot)
        reward, reward_info = self._calc_reward(action)
        terminated = not self.is_healthy # if robot is no longer healthy, terminate episode
        truncated = self._step >= (self._max_episode_time_sec / self.dt) # if time for episode has ended (not because robot failed)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        # only render in case enough time passed from last rendering
        if self.render_mode == "human" and (self.data.time - self._last_render_time) > (1.0 / self.metadata["render_fps"]):
            self.render()
            self._last_render_time = self.data.time

        self._last_action = action

        return observation, reward, terminated, truncated, info

    @property
    def is_healthy(self):
        """
        Property to define if the robot is in an healthy state (trunk in healthy range of position, roll and pitch are in the limits defined)
        """
        state = self.state_vector() # from MujocoEnv, the state used by the environment to check health of the robot during simulation
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z

        min_roll, max_roll = self._healthy_roll_range
        is_healthy = is_healthy and min_roll <= state[4] <= max_roll

        min_pitch, max_pitch = self._healthy_pitch_range
        is_healthy = is_healthy and min_pitch <= state[5] <= max_pitch

        return is_healthy

    @property
    def projected_gravity(self):
        """
        Property to project gravity with respect to the robot orientation
        """
        w, x, y, z = self.data.qpos[3:7]
        euler_orientation = np.array(self.euler_from_quaternion(w, x, y, z))
        projected_gravity_not_normalized = (np.dot(self._gravity_vector, euler_orientation) * euler_orientation)
        if np.linalg.norm(projected_gravity_not_normalized) == 0:
            return projected_gravity_not_normalized
        else:
            return projected_gravity_not_normalized / np.linalg.norm(projected_gravity_not_normalized)

    @property
    def proj_gravity(self):
        """
        Get the projected gravity vector acting on the robot
        (magnitude of the vector is still 1, since it is clipped on +-1)
        """
        base_id = self.model.body("base").id
        R_body_to_world = self.data.xmat[base_id].reshape(3, 3) # take the body to world transition matrix in a 3x3 format
        gravity_world = np.array([0.0, 0.0, -1.0]) # base gravity of the world
        return R_body_to_world.T @ gravity_world # transpose the body-to-world rotation matrix (obtain the world to body transformation) and multiply by the gravity of the world to get the gravity of the body

    @property
    def feet_contact_forces(self):
        """
        Property to get a single value for the feet contact with the ground
        """
        feet_contact_forces = self.data.cfrc_ext[self._cfrc_ext_feet_indices]
        return np.linalg.norm(feet_contact_forces, axis=1)

    ######### Positive Reward functions #########
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
    def heading_tracking_reward(self):
        # do not care about the target position (heading towards the objective position)
        pass

    @property
    def feet_air_time_reward(self):
        """Award strides depending on their duration only when the feet makes contact with the ground"""
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        # if feet_air_time is > 0 (feet was in the air) and contact_filter detects a contact with the ground
        # then it is the first contact of this stride
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = np.sum((self._feet_air_time - 1.0) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= np.linalg.norm(self._desired_velocity[:2]) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        return air_time_reward

    @property
    def healthy_reward(self):
        return self.is_healthy

    ######### Negative Reward functions #########
    @property  # TODO: Not used
    def feet_contact_forces_cost(self):
        return np.sum(
            (self.feet_contact_forces - self._max_contact_force).clip(min=0.0)
        )

    @property
    def non_flat_base_cost(self):
        # Penalize the robot for not being flat on the ground
        return np.sum(np.square(self.proj_gravity[:2]))

    @property
    def collision_cost(self):
        # Penalize collisions on selected bodies
        return np.sum(
            1.0
            * (np.linalg.norm(self.data.cfrc_ext[self._cfrc_ext_contact_indices]) > 0.1)
        )

    @property
    def joint_limit_cost(self):
        # Penalize the robot for joints exceeding the soft control range
        out_of_range = (self._soft_joint_range[:, 0] - self.data.qpos[7:]).clip(
            min=0.0
        ) + (self.data.qpos[7:] - self._soft_joint_range[:, 1]).clip(min=0.0)
        return np.sum(out_of_range)

    @property
    def torque_cost(self):
        # Last 12 values are the motor torques
        return np.sum(np.square(self.data.qfrc_actuator[-12:]))

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
    def default_joint_position_cost(self):
        """
        Penalizes the distance from the base configuration of the joints
        """
        return np.sum(np.square(self.data.qpos[7:] - self._default_joint_position))

    @property
    def smoothness_cost(self):
        return np.sum(np.square(self.data.qpos[7:] - self._last_action))

    

    def _calc_reward(self, action):
        # TODO: Add debug mode with custom Tensorboard calls for individual reward
        #   functions to get a better sense of the contribution of each reward function

        # Positive Rewards
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
        rewards = (
            linear_vel_tracking_reward
            + angular_vel_tracking_reward
            + healthy_reward
            + feet_air_time_reward
        )

        # Negative Costs
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
        default_joint_position_cost = (
            self.default_joint_position_cost
            * self.cost_weights["default_joint_position"]
        )
        costs = (
            ctrl_cost
            + action_rate_cost
            + vertical_vel_cost
            + xy_angular_vel_cost
            + joint_limit_cost
            + joint_acceleration_cost
            + orientation_cost
            + default_joint_position_cost
        )

        reward = max(0.0, rewards - costs)
        # reward = rewards - self.curriculum_factor * costs
        reward_info = {
            "linear_vel_tracking_reward": linear_vel_tracking_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

    def _get_obs(self):
        # The first three indices are the global x,y,z position of the trunk of the robot
        # The second four are the quaternion representing the orientation of the robot
        # The above seven values are ignored since they are privileged information
        # The remaining 12 values are the joint positions
        dofs_position = self.data.qpos[7:] - self._default_joint_position # The joint positions are relative to the starting position

        # The first three values are the global linear velocity of the robot
        # The second three are the angular velocity of the robot
        # The remaining 12 values are the joint velocities
        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3]
        base_angular_velocity = velocity[3:6]
        dofs_velocity = velocity[6:]

        desired_vel = self._desired_velocity
        last_action = self._last_action
        projected_gravity = self.proj_gravity

        curr_obs = np.concatenate(
            (
                base_linear_velocity * self._obs_scale["linear_velocity"],
                base_angular_velocity * self._obs_scale["angular_velocity"],
                projected_gravity,
                desired_vel * self._obs_scale["linear_velocity"],
                dofs_position * self._obs_scale["dofs_position"],
                dofs_velocity * self._obs_scale["dofs_velocity"],
                last_action, # to produce a smoother sequence of commands (and we penalize the difference with respect to the previous action taken by the policy)
            )
        ).clip(-self._clip_obs_threshold, self._clip_obs_threshold)

        print(curr_obs.shape)

        return curr_obs

    def reset_model(self):
        # Reset the position and control values with noise
        self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        self.data.ctrl[:] = self.model.key_ctrl[
            0
        ] + self._reset_noise_scale * self.np_random.standard_normal(
            *self.data.ctrl.shape
        )

        # Reset the variables and sample a new desired velocity
        self._desired_velocity = self._sample_desired_vel()
        self._step = 0
        self._last_action = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def _sample_desired_vel(self):
        desired_vel = np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
        return desired_vel

    @staticmethod
    def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians