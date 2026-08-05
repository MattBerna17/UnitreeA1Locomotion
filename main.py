import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # 1. Caricamento del Modello Fisico (scena)
    # Punta al file XML appena scaricato
    xml_path = "unitree_a1/scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 2. Configurazione della Posa di Base (Standing)
    # L'A1 ha 12 motori. Ordine: FR, FL, RR, RL (Abad, Hip, Knee)
    # Una buona posa stabile per stare in piedi:
    target_angles = np.array([0.0, 0.9, -1.8] * 4)

    # Inizializziamo il robot già sollevato da terra per evitare che compenetri il pavimento
    # qpos[0:7] = [x, y, z, qw, qx, qy, qz] (Posizione e orientamento del tronco nel mondo)
    # 4 rotazioni (quaternione): altrimenti, usando qx, qy, qz andremo incontro al problema del gimbal lock: se ruotiamo in una certa combinazione, arriviamo a una configurazione in cui due assi si sovrappongono e quindi perdiamo un grado di libertà (inoltre l'ordine di rotazione conta: https://www.youtube.com/watch?v=BczeMqU_u2Y)
    # usando il quaternione, abbiamo qx, qy, qz assi definiti e qw scalare di quanto dobbiamo ruotare su quell'asse
    # con la velocità, il problema non si pone perché si tratta di movimenti istantanei, non di pose assolute
    data.qpos[:7] = [0.0, 0.0, 0.32, 1.0, 0.0, 0.0, 0.0] # initial z: 0.32 because 0.05 is the ground of the scene and 0.27 is the z value of the hip
    # qpos[7:] = angoli dei 12 giunti
    data.qpos[7:] = target_angles

    # 3. Parametri del Controllore PD
    # Simuliamo la rigidità dei giunti per farlo stare in piedi contro la gravità
    kp = 60.0  # Guadagno Proporzionale (Rigidità)
    kd = 1.5   # Guadagno Derivativo (Smorzamento)

    # Eseguiamo un primo step di calcolo cinematico per aggiornare le matrici interne
    mujoco.mj_forward(model, data)

    print("Avvio simulazione. Chiudi la finestra del viewer per terminare.")

    # 4. Avvio del Viewer Passivo e Loop di Simulazione
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            # --- CONTROLLORE PD ---
            # Leggiamo lo stato attuale dai sensori simulati
            current_angles = data.qpos[7:]  
            current_velocities = data.qvel[6:] # In qvel, le velocità spaziali sono 6 (non 7 come i quaternioni)

            # Calcoliamo l'errore (Legge di controllo: Torque = kp*err_pos - kd*vel)
            torques = kp * (target_angles - current_angles) - (kd * current_velocities)
            
            # Inviamo le coppie calcolate agli attuatori
            data.ctrl[:] = torques

            # --- STEP FISICO ---
            mujoco.mj_step(model, data)

            # --- SINCRONIZZAZIONE VISIVA ---
            viewer.sync()

            # Manteniamo la simulazione in tempo reale
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()