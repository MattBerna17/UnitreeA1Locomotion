from env import Environment
from env1 import A1Env

def main():
    # Chiediamo esplicitamente di aprire la finestra 3D
    env = A1Env(render_mode="human")
    obs, info = env.reset()

    print("Simulazione Gym Custom avviata! Premi Ctrl+C per uscire.")

    try:
        while True:
            # Genera 12 valori a caso e inviali all'ambiente
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nChiusura in corso...")
    finally:
        env.close()

if __name__ == "__main__":
    main()