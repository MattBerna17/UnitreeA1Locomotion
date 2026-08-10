import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def main():
    # 1. load the scene for the unitree a1 model
    xml_path = "unitree_a1/scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 2. base configuration
    # a1 has 12 motors: FR, FL, RR, RL (abad, hip, knee)
    target_angles = np.array([0.0, 0.9, -1.8] * 4) # standing position

    # initialize robot above the ground
    # qpos[0:7] = [x, y, z, qw, qx, qy, qz] (position and orientation of the start)
    # 4 rotazioni (quaternione): altrimenti, usando qx, qy, qz andremo incontro al problema del gimbal lock: se ruotiamo in una certa combinazione, arriviamo a una configurazione in cui due assi si sovrappongono e quindi perdiamo un grado di libertà (inoltre l'ordine di rotazione conta: https://www.youtube.com/watch?v=BczeMqU_u2Y)
    # usando il quaternione, abbiamo qx, qy, qz assi definiti e qw scalare di quanto dobbiamo ruotare su quell'asse
    # con la velocità, il problema non si pone perché si tratta di movimenti istantanei, non di pose assolute
    data.qpos[:7] = [0.0, 0.0, 0.32, 1.0, 0.0, 0.0, 0.0] # initial z: 0.32 because 0.05 is the ground of the scene and 0.27 is the z value of the hip
    # qpos[7:] = angles of the 12 joints
    data.qpos[7:] = target_angles

    # 3. PD controller params
    kp = 60.0  # proportional (rigidity)
    kd = 1.5   # derivative (smoothness)

    # first step of forward kinematics
    mujoco.mj_forward(model, data)

    print("Simulation start. Close window to end simulation.")

    # 4. start visual simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            # read sensors of the current step
            current_angles = data.qpos[7:] # current positions and orientations
            current_velocities = data.qvel[6:] # current velocities

            # compute error: torque = kp*err_pos - kd*vel
            torques = kp * (target_angles - current_angles) - (kd * current_velocities)
            torques = np.random.randn(12) # random movement
            # print(torques)
            
            # send commands to the controllers
            data.ctrl[:] = torques

            # execute step for the model with commands passed
            mujoco.mj_step(model, data)



            # visualize the step
            viewer.sync()
            # real time simulation
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()