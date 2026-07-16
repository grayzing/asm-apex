import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# IEEE SINGLE-COLUMN CONFIGURATION
# -------------------------------------------------------------------------
IEEE_SINGLE_COLUMN_WIDTH = 3.5  # Exact column width in inches
aspect_ratio = 0.72            # Compact landscape ratio for a single column
fig_width = IEEE_SINGLE_COLUMN_WIDTH
fig_height = fig_width * aspect_ratio

plt.rcParams.update({
    "text.usetex": False,            # Set to True if local LaTeX is installed
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.0,                # Matches IEEE caption/label size
    "axes.labelsize": 8.5,           
    "axes.titlesize": 8.5,           
    "xtick.labelsize": 7.5,          
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": True,
    "legend.fancybox": False,        # Clean sharp-corner frame
    "legend.edgecolor": "black",     # Classic thin black border
    "axes.linewidth": 0.5,           # Thin clean axis borders
    "grid.linewidth": 0.3,           # Subtle grid line weight
    "grid.linestyle": ":",           # Dotted grid matching the results plot
    "lines.linewidth": 1.0,          
})

def plot_reward_curve(csv_path="training_log.csv", save_path="reward_curve.pdf", window_size=50):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"'{csv_path}' not found. Generating realistic dummy training data...")
        # Fallback dummy data generation
        np.random.seed(42)
        episodes = np.arange(1, 4800)
        # S-curve ascent representing typical reinforcement learning progress
        base_reward = 1400 + 430 * (1 - np.exp(-episodes / 1200))
        noise = np.random.normal(0, 45, len(episodes))
        # Periodic exploration drops
        exploration_drops = -120 * (np.sin(episodes / 150) ** 4) * np.exp(-episodes / 2500)
        raw_rewards = base_reward + noise + exploration_drops
        df = pd.DataFrame({'episode': episodes, 'reward': raw_rewards})

    # Group by episode to compute mean episodic reward
    episode_data = df.groupby('episode')['reward'].mean().reset_index()
    episodes = episode_data['episode'].values
    rewards = episode_data['reward'].values
    
    # Calculate rolling window average
    if len(rewards) > window_size:
        smoothed_rewards = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean().values
    else:
        smoothed_rewards = rewards

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=600)
    ax.grid(True, zorder=0)

    # 1. Plot highly volatile raw reward (faded blue, rasterized to keep PDF sizes low)
    ax.plot(episodes, rewards, color='#aec7e8', alpha=0.45, label='Raw Reward', rasterized=True, zorder=1)
    
    # 2. Plot smoothed rolling average (solid high-contrast blue line)
    ax.plot(episodes, smoothed_rewards, color='#1f77b4', alpha=1.0, label=f'Moving Avg. (N={window_size})', zorder=2)

    # Label styling
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Cumulative Reward')
    ax.set_xlim(episodes.min(), episodes.max())
    
    # Matching ticks facing inside
    ax.tick_params(direction='in', length=3, width=0.5, top=True, right=True)
    
    # Opaque legend background so raw data lines don't strike through text
    ax.legend(loc='lower right', framealpha=1.0, facecolor='white', edgecolor='0.8')

    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"Successfully saved Reward Curve to: {save_path}")

if __name__ == "__main__":
    plot_reward_curve()