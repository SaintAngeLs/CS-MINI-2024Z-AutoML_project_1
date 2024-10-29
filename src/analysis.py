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

    # Adding labels to points
    for i, row in history_df.iterrows():
        plt.text(row['iteration'], row['grid_score'], f'{row["grid_score"]:.3f}', fontsize=9, ha='right')
        plt.text(row['iteration'], row['random_score'], f'{row["random_score"]:.3f}', fontsize=9, ha='left')
        plt.text(row['iteration'], row['bayes_score'], f'{row["bayes_score"]:.3f}', fontsize=9, ha='center')

    # Titles and labels
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Ensure that the "../results" directory exists, if not, create it
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the plot to the results directory
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()
