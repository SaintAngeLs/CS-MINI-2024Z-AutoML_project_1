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



class CriticalDifferencePlot:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def wilcoxon_holm(self, rankings):
        """
        Applies the Wilcoxon signed rank test between each pair of classifiers and 
        uses Holm correction to reject null hypotheses.
        """
        classifiers = list(rankings.keys())
        p_values = []
        num_classifiers = len(classifiers)

        # Wilcoxon tests between all classifier pairs
        for i in range(num_classifiers - 1):
            for j in range(i + 1, num_classifiers):
                classifier_1 = classifiers[i]
                classifier_2 = classifiers[j]
                ranks_1 = rankings[classifier_1]
                ranks_2 = rankings[classifier_2]

                # Ensure there are ranks to compare
                if len(ranks_1) > 0 and len(ranks_2) > 0:
                    p_value = stats.wilcoxon(ranks_1, ranks_2)[1]
                    p_values.append((classifier_1, classifier_2, p_value, False))

        # Holm correction
        k = len(p_values)
        p_values.sort(key=lambda x: x[2])  # Sort by p-value
        for i in range(k):
            new_alpha = self.alpha / (k - i)
            if p_values[i][2] <= new_alpha:
                p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
            else:
                break

        return p_values

    def calculate_average_ranks(self, rankings):
        """
        Calculates the average ranks for each classifier based on its rankings across datasets.
        Filters out NaN or empty rankings to avoid errors.
        """
        avg_ranks = {}
        for model, ranks in rankings.items():
            # Check if there are valid rankings to compute the mean
            if len(ranks) > 0:
                avg_rank = np.nanmean(ranks)  # Handle any NaN values gracefully
                if not np.isnan(avg_rank):
                    avg_ranks[model] = avg_rank
        return avg_ranks

    def form_cliques(self, p_values, classifiers):
        """
        Group classifiers into cliques based on non-significant pairwise comparisons.
        """
        g = nx.Graph()
        g.add_nodes_from(range(len(classifiers)))
        for p_value in p_values:
            if not p_value[3]:
                i = classifiers.index(p_value[0])
                j = classifiers.index(p_value[1])
                g.add_edge(i, j)
        return list(nx.find_cliques(g))

    def plot(self, avg_ranks, classifiers, p_values, filename, title=None):
        """
        Generates and saves the critical difference plot. Handles cases where
        average ranks are missing or invalid.
        """
        if len(avg_ranks) == 0:
            print("No valid rankings available to plot.")
            return

        # Safely compute low and high values for the ranks
        try:
            lowv = min(1, int(np.floor(min(avg_ranks.values()))))
            highv = max(len(avg_ranks), int(np.ceil(max(avg_ranks.values()))))
        except ValueError:
            print("Error calculating ranks: No valid values available.")
            return

        width = 9
        textspace = 1.5

        fig = plt.figure(figsize=(width, 6))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        # Scale factors
        scalewidth = width - 2 * textspace

        def rankpos(rank):
            return textspace + scalewidth / (highv - lowv) * (rank - lowv)

        # Draw the main comparison line
        cline = 0.4
        ax.plot([0, 1], [cline, cline], c='k', linewidth=2)
        bigtick = 0.3
        smalltick = 0.15

        # Draw ticks for ranks
        for a in np.arange(lowv, highv + 0.5, 0.5):
            tick = smalltick if a % 1 else bigtick
            ax.plot([rankpos(a), rankpos(a)], [cline - tick / 2, cline + tick / 2], c='k', lw=2)

        # Draw rank labels
        for i in range(lowv, highv + 1):
            ax.text(rankpos(i), cline - 0.4 * bigtick, str(i), ha="center", va="center", size=10)

        # Draw classifier names and their average ranks
        for idx, (name, rank) in enumerate(sorted(avg_ranks.items(), key=lambda x: x[1])):
            y_pos = cline + 0.4 + idx * 0.3
            ax.text(rankpos(rank), y_pos, name, ha="center", va="center", size=12)
            ax.plot([rankpos(rank), rankpos(rank)], [cline, y_pos], c='k', lw=2)

        # Draw significance cliques
        cliques = self.form_cliques(p_values, list(avg_ranks.keys()))
        start_y = cline + (len(classifiers) + 1) * 0.3
        for clique in cliques:
            if len(clique) == 1:
                continue
            min_idx = min(clique)
            max_idx = max(clique)
            ax.plot([rankpos(avg_ranks[classifiers[min_idx]]), rankpos(avg_ranks[classifiers[max_idx]])], [start_y, start_y], c='k', lw=4)
            start_y += 0.3

        if title:
            plt.title(title, fontsize=14)
        plt.savefig(filename)
        plt.close()