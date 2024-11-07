import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import networkx as nx
from scipy import stats


def plot_results(history, title="Tuning Results", filename="plot.png"):
    history_df = pd.DataFrame(history)
    plt.figure(figsize=(10, 6))
    
    # Plot individual lines for each method
    plt.plot(history_df['iteration'], history_df['grid_score'], label='Grid Search', marker='o', linestyle='--')
    plt.plot(history_df['iteration'], history_df['random_score'], label='Random Search', marker='s', linestyle=':')
    plt.plot(history_df['iteration'], history_df['bayes_score'], label='Bayesian Optimization', marker='^', linestyle='-')

    # Adding labels to points with condition to avoid clutter
    for i, row in history_df.iterrows():
        if not pd.isna(row["grid_score"]):
            plt.text(row['iteration'], row['grid_score'], f'{row["grid_score"]:.4f}', fontsize=8, ha='right')
        if not pd.isna(row["random_score"]):
            plt.text(row['iteration'], row['random_score'], f'{row["random_score"]:.4f}', fontsize=8, ha='left')
        if not pd.isna(row["bayes_score"]):
            plt.text(row['iteration'], row['bayes_score'], f'{row["bayes_score"]:.4f}', fontsize=8, ha='center')

    # Titles and labels
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()