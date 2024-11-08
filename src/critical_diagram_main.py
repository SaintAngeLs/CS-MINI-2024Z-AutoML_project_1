import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import wilcoxon, friedmanchisquare
import operator

# Set up matplotlib to use non-interactive backend
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

def load_and_prepare_data(data_path):
    # Load the dataset into a DataFrame and inspect column names
    df = pd.read_csv(data_path)
    print("Data loaded with columns:", df.columns)
    return df

# Function to form cliques based on p-values using Holm correction
def form_cliques(p_values, classifiers):
    m = len(classifiers)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if not p[3]:  # Not significant
            i = classifiers.index(p[0])
            j = classifiers.index(p[1])
            g_data[min(i, j), max(i, j)] = 1

    g = nx.Graph(g_data)
    return list(nx.find_cliques(g))


def graph_ranks(avranks, classifiers, p_values, cd=None, reverse=False, width=10, textspace=1.5):
    """
    Basic CD diagram plotting function for classifier performance comparison.
    This implementation is simple and may not include all the aesthetics of Orange3's graph_ranks.
    """
    n_classifiers = len(classifiers)
    fig, ax = plt.subplots(figsize=(width, n_classifiers / 2))

    sorted_classifiers = sorted(avranks.items(), key=lambda x: x[1], reverse=reverse)
    classifier_names, ranks = zip(*sorted_classifiers)

    # Plot ranks
    ax.plot(ranks, range(len(classifier_names)), 'o')
    for i, (name, rank) in enumerate(sorted_classifiers):
        ax.text(rank + textspace, i, name, verticalalignment='center')

    ax.set_xlim([0, max(ranks) + textspace * 2])
    ax.set_ylim([-1, len(classifiers)])

    # Plot Critical Difference (CD) line if provided
    if cd:
        ax.hlines(y=len(classifiers) - 1, xmin=min(ranks) - cd / 2, xmax=min(ranks) + cd / 2, color='r', label='CD')
        ax.legend()

    ax.invert_yaxis()
    ax.set_xlabel("Average Rank")
    plt.tight_layout()
    plt.show()

def draw_cd_diagram(df, dataset_name, alpha=0.05):
    # Filter data for a specific dataset
    df_dataset = df[df['dataset'] == dataset_name]
    print(f"Processing dataset: {dataset_name}")
    
    # Check if 'score' column exists in the filtered DataFrame
    if 'best_score' not in df_dataset.columns:
        print(f"Warning: 'score' column not found in dataset '{dataset_name}'. Skipping.")
        return
    
    classifiers = df_dataset['model'].unique()

    # Prepare the accuracy scores for Friedman test
    scores = [df_dataset[df_dataset['model'] == model]['best_score'].values for model in classifiers]

    # Apply Friedman's test to check overall significance
    friedman_p_value = friedmanchisquare(*scores).pvalue
    if friedman_p_value >= alpha:
        print(f"No significant differences found for {dataset_name}")
        return

   # Wilcoxon-Holm corrected p-values for pairwise comparisons
    p_values = []
    for i in range(len(classifiers) - 1):
        for j in range(i + 1, len(classifiers)):
            if np.array_equal(scores[i], scores[j]):
                # Identical scores, set p-value to 1.0
                p_value = 1.0
            else:
                # Perform Wilcoxon test if scores differ
                p_value = wilcoxon(scores[i], scores[j]).pvalue
            p_values.append((classifiers[i], classifiers[j], p_value, False))


    # Holm correction
    k = len(p_values)
    p_values.sort(key=operator.itemgetter(2))
    for i in range(k):
        new_alpha = alpha / (k - i)
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            break

    # Calculate average ranks
    df_ranks = pd.DataFrame([df_dataset[df_dataset['model'] == model]['best_score'].rank(ascending=False) for model in classifiers], index=classifiers).mean(axis=1).sort_values()

    # Plot CD Diagram
    avranks = df_ranks.to_dict()
    graph_ranks(avranks, classifiers, p_values, cd=None, reverse=True, width=9, textspace=1.5)
    plt.title(f"CD Diagram for {dataset_name}")
    plt.savefig(f'cd-diagram-{dataset_name}.png', bbox_inches='tight')
    plt.close()

# Main function to process each dataset
def main(data_path):
    df = load_and_prepare_data(data_path)
    for dataset_name in df['dataset'].unique():
        draw_cd_diagram(df, dataset_name)

main("../results/best_tuning_results.csv")
