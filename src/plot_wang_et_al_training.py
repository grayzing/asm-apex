import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results(csv_file='training_log_wang_et_al.csv', window_size=10):
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Group by episode to get the mean loss and reward per episode
    # (Since you are logging every transition, you have many entries per episode)
    episode_df = df.groupby('episode').agg({
        'loss': 'mean',
        'reward': 'mean'
    }).reset_index()

    # Apply rolling mean
    episode_df['loss_smooth'] = episode_df['loss'].rolling(window=window_size).mean()
    episode_df['reward_smooth'] = episode_df['reward'].rolling(window=window_size).mean()

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Reward
    ax1.plot(episode_df['episode'], episode_df['reward'], alpha=0.3, color='blue')
    ax1.plot(episode_df['episode'], episode_df['reward_smooth'], color='blue', label='Moving Avg')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Progress')
    ax1.legend()

    # Plot Loss
    ax2.plot(episode_df['episode'], episode_df['loss'], alpha=0.3, color='red')
    ax2.plot(episode_df['episode'], episode_df['loss_smooth'], color='red', label='Moving Avg')
    ax2.set_ylabel('Average Loss')
    ax2.set_xlabel('Episode')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.show()

if __name__ == "__main__":
    plot_training_results()