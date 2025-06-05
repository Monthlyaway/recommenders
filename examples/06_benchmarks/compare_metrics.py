import os
import pandas as pd
import numpy as np
import proplot as pplt
import matplotlib.pyplot as plt
import seaborn as sns

# --- Original Settings and Data Loading (largely unchanged) ---
# Radar chart settings (Label dictionaries)
ranking_dimensions_dict = {
    'MAP': 'MAP',
    'nDCG@10': 'nDCG@10',
    'Precision@10': 'Precision@10',
    'Recall@10': 'Recall@10',
}
# Keys used for data extraction for ranking metrics
ranking_metric_keys = ['MAP', 'nDCG@k', 'Precision@k', 'Recall@k']

rating_dimensions_dict = {
    'RMSE': 'RMSE\n(inverted)',
    'MAE': 'MAE\n(inverted)',
    'R2': 'R²',
    'EVS': 'EVS',
    'Training_Efficiency': 'Training Time\n(inverted)'
}
# Keys used for data extraction for rating metrics (before transformation)
rating_metric_keys = ['RMSE', 'MAE', 'R2',
                      'Explained Variance', 'Train time (s)']

# File and folder setup
try:
    BASE_DIR = os.path.dirname(__file__)
except NameError:
    BASE_DIR = "."
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Algorithm list
algos = ['als', 'svd', 'sar', 'ncf', 'bpr', 'lightgcn']

# Read all result files for 1m dataset
algo_results = {}
for algo in algos:
    file_path = os.path.join(RESULTS_DIR, f'{algo}_1m.csv')
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                algo_results[algo] = df.iloc[-1]
            else:
                print(f"Warning: {file_path} is empty.")
        except pd.errors.EmptyDataError:
            print(f"Warning: {file_path} is empty or invalid.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    else:
        print(f"Warning: {file_path} not found.")

# --- 1. Ranking Metrics Data Preparation ---
ranking_algo_data = {}
for algo in algos:
    if algo in algo_results:
        row = algo_results[algo]
        current_algo_data = [row.get(key, np.nan)
                             for key in ranking_metric_keys]
        if not any(np.isnan(current_algo_data)):
            ranking_algo_data[algo.upper()] = current_algo_data
        else:
            print(f"Skipping {algo} for ranking chart due to missing data.")

# --- 2. Rating Metrics Data Preparation ---
rating_algo_data = {}
for algo in algos:
    if algo in algo_results:
        row = algo_results[algo]
        current_algo_data = [
            row.get('RMSE', np.nan), row.get('MAE', np.nan),
            row.get('R2', np.nan), row.get('Explained Variance', np.nan),
            row.get('Train time (s)', np.nan)
        ]
        if not any(np.isnan(current_algo_data)):
            # Store original values for display
            original_values = current_algo_data.copy()
            # Invert RMSE, MAE, and Training time for visualization
            current_algo_data[0] = 1.0 / (current_algo_data[0] + 1e-9)  # RMSE
            current_algo_data[1] = 1.0 / (current_algo_data[1] + 1e-9)  # MAE
            current_algo_data[4] = 1.0 / \
                (current_algo_data[4] + 1e-9)  # Training time
            rating_algo_data[algo.upper()] = (
                current_algo_data, original_values)
        else:
            print(f"Skipping {algo} for rating chart due to missing data.")

# --- Create Seaborn/Matplotlib Bar Charts ---
if len(ranking_algo_data) == 0 and len(rating_algo_data) == 0:
    print("No data available for any plots. Figure not generated.")
else:
    # Define a visually distinct color palette (blue/grey inspired by test.py)
    fancy_colors = [
        '#E8F0FE',  # very light blue
        '#9E9E9E',  # grey
        '#8AB4F8',  # light blue
        '#C0C0C0',  # light grey
        '#FAC858',  # gold 
        '#4285F4',  # blue
    ]
    all_algos = sorted(set(list(ranking_algo_data.keys()) +
                       list([k for k in rating_algo_data.keys()])))
    color_dict = {algo: fancy_colors[i % len(
        fancy_colors)] for i, algo in enumerate(all_algos)}

    fig, axs = plt.subplots(2, 1, figsize=(14, 10))

    # --- Plot 1: Ranking Metrics ---
    if ranking_algo_data:
        ax = axs[0]
        metric_labels = [ranking_dimensions_dict.get(
            key.replace('@k', '@10'), key) for key in ranking_metric_keys]
        df_ranking = pd.DataFrame(ranking_algo_data, index=metric_labels).T.reset_index(
        ).rename(columns={'index': 'Algorithm'})
        df_ranking_melted = df_ranking.melt(
            id_vars=['Algorithm'], var_name='Metric', value_name='Score')
        sns.barplot(x='Metric', y='Score', hue='Algorithm',
                    data=df_ranking_melted, ax=ax, palette=color_dict)
        # Add value labels
        for p in ax.patches:
            if not pd.isna(p.get_height()):
                ax.annotate(f'{p.get_height():.3f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 10),
                            textcoords='offset points', fontsize=8)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('')
        ax.set_title('Ranking Metrics Comparison', fontsize=14)
        ax.set_ylim(0, 0.5)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Centered legend at the top
        ax.legend(title='', loc='upper center', bbox_to_anchor=(
            0.5, 1.15), ncol=len(ranking_algo_data), frameon=True, fontsize=10)

    # --- Plot 2: Rating Metrics ---
    if rating_algo_data:
        ax = axs[1]
        metric_labels = list(rating_dimensions_dict.values())
        # Prepare DataFrame for plotting
        rating_plot_data = []
        for algo, (values, original) in rating_algo_data.items():
            for i, (val, orig) in enumerate(zip(values, original)):
                rating_plot_data.append({
                    'Algorithm': algo,
                    'Metric': metric_labels[i],
                    'Score': val,
                    'Original': orig
                })
        df_rating = pd.DataFrame(rating_plot_data)
        sns.barplot(x='Metric', y='Score', hue='Algorithm',
                    data=df_rating, ax=ax, palette=color_dict)
        # Add value labels (show original values for RMSE, MAE, Training time)
        for p in ax.patches:
            if not pd.isna(p.get_height()):
                ax.annotate(f'{p.get_height():.3f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 10),
                            textcoords='offset points', fontsize=8)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylim(0, 1.7)
        ax.set_title('Rating Metrics Comparison', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        # Centered legend at the top
        ax.legend(title='', loc='upper center', bbox_to_anchor=(
            0.5, 1.15), ncol=len(rating_algo_data), frameon=True, fontsize=10)

    fig.suptitle(
        'Algorithm Performance Comparison (MovieLens 1M Dataset)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = os.path.join(FIGURES_DIR, 'seaborn_bar_comparison.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Seaborn bar chart saved to {save_path}")
    # plt.show()  # Uncomment to display the plot
