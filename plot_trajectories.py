import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_comparison_dashboard(csv_path="trajectory_tracking_results.csv", trajectory_name="circle"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR]: File {csv_path} not found.")
        return

    df_traj = df[df["trajectory"] == trajectory_name]
    
    if df_traj.empty:
        print(f"No data found for trajectory: {trajectory_name}")
        return
        
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Tracking Performance: PPO vs SAC ({trajectory_name.capitalize()})", fontsize=16, fontweight='bold', y=1.05)
    
    # 1. Traiettoria XY (top-down view)
    sns.scatterplot(data=df_traj, x="x", y="y", hue="algorithm", s=15, edgecolor=None, ax=axs[0], palette="Set1")
    axs[0].set_title("Top-Down Path")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].axis("equal")
    
    # 2. Cross-Track Error (precision)
    sns.lineplot(data=df_traj, x="time_s", y="cross_track_error", hue="algorithm", ax=axs[1], palette="Set1")
    axs[1].set_title("Cross-Track Error")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Distance from Target (m)")
    
    # 3. Wz Command (stability of controller)
    sns.lineplot(data=df_traj, x="time_s", y="wz_command", hue="algorithm", ax=axs[2], palette="Set1", alpha=0.8)
    axs[2].set_title("Angular Velocity Command (wz)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Command (rad/s)")
    
    plt.tight_layout()
    filename = f"tracking_comparison_{trajectory_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {filename}")
    plt.close()

if __name__ == "__main__":
    for traj in ["line", "cosine", "circle"]:
        generate_comparison_dashboard(trajectory_name=traj)