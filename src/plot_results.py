import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_simulation_results(csv_file):
    # Load the data
    df = pd.read_csv(csv_file)

    # Configuration for re-labeling
    method_order = ["Random", "Normal", "Enhanced", "Basic", "VDN"]
    new_labels = ["Random", "ALL-ON", "Liang et al. xApp 2", "Liang et al. xApp 1", "VDN"]
    
    # 1. Box and Whisker Plot: Sleeping Sectors
    plt.figure(figsize=(10, 6))
    plot_data = [df[df['Method'] == m]['Avg_Sleeping_Sectors'] for m in method_order]
    
    plt.boxplot(plot_data, labels=new_labels, patch_artist=True)
    plt.title("Average Number of Sleeping Sectors per Method")
    plt.ylabel("Average Number of Sleeping Sectors")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('sleeping_sectors_plot.pdf') # Saving as PDF for vector quality
    plt.show()

    # 2. CDF Plots: Throughput
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6))

    for m, label in zip(method_order, new_labels):
        # 10th Percentile Throughput
        data_p10 = sorted(df[df['Method'] == m]['P10_Throughput'])
        cdf_p10 = np.linspace(0, 1, len(data_p10))
        ax2.plot(data_p10, cdf_p10, marker='o', markersize=4, label=label)
        
        # Average Throughput
        data_avg = sorted(df[df['Method'] == m]['Avg_Throughput'])
        cdf_avg = np.linspace(0, 1, len(data_avg))
        ax3.plot(data_avg, cdf_avg, marker='o', markersize=4, label=label)

    # Formatting CDFs
    for ax, title in zip([ax2, ax3], ["CDF: 10th Percentile Throughput", "CDF: Average Throughput"]):
        ax.set_title(title)
        ax.set_xlabel("Throughput (Mbps)")
        ax.set_ylabel("CDF")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('throughput_cdf_plot.pdf') # Saving as PDF for vector quality
    plt.show()

if __name__ == "__main__":
    # Ensure your CSV file is in the same directory as this script
    plot_simulation_results('simulation_results/simulation_results_19gnb.csv')