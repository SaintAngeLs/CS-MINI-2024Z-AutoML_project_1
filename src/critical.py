import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx
from scipy.stats import wilcoxon, friedmanchisquare
import math
import operator

# Set up matplotlib to use non-interactive backend
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

# Function to draw the critical difference (CD) diagram
def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False):
    """
    Draws a Critical Difference (CD) graph to display the differences in methods' performance.
    """
    width = float(width)
    textspace = float(textspace)
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            return textspace + scalewidth / (highv - lowv) * (rank - lowv)
        else:
            return textspace + scalewidth / (highv - lowv) * (highv - rank)

    sums = avranks  # Expecting avranks to be a dictionary
    nnames = names

    if lowv is None:
        lowv = min(1, int(math.floor(min(sums.values()))))
    if highv is None:
        highv = max(len(sums), int(math.ceil(max(sums.values()))))

    cline = 0.4
    distanceh = 0.25
    cline += distanceh
    minnotsignificant = max(2 * 0.2, 0)
    height = cline + ((len(sums) + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    hf = 1. / height
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Plot main comparison line
    ax.plot([0, 1], [cline, cline], c='k', linewidth=2)
    bigtick = 0.3
    smalltick = 0.15

    # Plot ticks for ranks
    for a in np.arange(lowv, highv + 0.5, 0.5):
        tick = smalltick if a % 1 else bigtick
        ax.plot([rankpos(a), rankpos(a)], [cline - tick / 2, cline + tick / 2], c='k', lw=2)

    # Plot rank labels
    for i in range(lowv, highv + 1):
        ax.text(rankpos(i), cline - 0.4 * bigtick, str(i), ha="center", va="center", size=10)

    # Plot classifiers and their ranks
    for idx, (name, rank) in enumerate(sorted(avranks.items(), key=lambda x: x[1])):
        y_pos = cline + 0.4 + idx * 0.3
        ax.text(rankpos(rank), y_pos, name, ha="center", va="center", size=12)
        ax.plot([rankpos(rank), rankpos(rank)], [cline, y_pos], c='k', lw=2)

    # Draw cliques for non-significant differences
    start_y = cline + (len(names) + 1) * 0.3
    cliques = form_cliques(p_values, list(avranks.keys()))
    for clique in cliques:
        if len(clique) == 1:
            continue
        min_idx = min(clique)
        max_idx = max(clique)
        ax.plot([rankpos(avranks[names[min_idx]]), rankpos(avranks[names[max_idx]])], [start_y, start_y], c='k', lw=4)
        start_y += 0.3

    if filename:
        plt.savefig(filename)
    plt.close()


def form_cliques(p_values, nnames):
    """
    Form cliques based on Wilcoxon-Holm corrected p-values.
    """
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if not p[3]:
            i = nnames.index(p[0])
            j = nnames.index(p[1])
            g_data[min(i, j), max(i, j)] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)

def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False):
    """
    Draws a critical difference diagram based on performance data and Wilcoxon-Holm correction.
    """
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)
    graph_ranks(average_ranks.values, average_ranks.keys(), p_values, cd=None, reverse=True, width=9, textspace=1.5, labels=labels)

    if title:
        plt.title(title, fontsize=22)
    plt.savefig('cd-diagram.png', bbox_inches='tight')

def wilcoxon_holm(alpha=0.05, df_perf=None):
    """
    Applies the Wilcoxon signed-rank test between each pair of algorithms and applies Holm correction.
    """
    df_counts = pd.DataFrame({'count': df_perf.groupby(['classifier_name']).size()}).reset_index()
    max_nb_datasets = df_counts['count'].max()
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]['classifier_name'])

    # Test the null hypothesis using Friedman's test
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy']) for c in classifiers
    ))[1]
    
    if friedman_p_value >= alpha:
        print('The null hypothesis over the entire classifiers cannot be rejected')
        return [], pd.Series(), 0

    # Wilcoxon signed-rank tests and Holm correction
    p_values = []
    m = len(classifiers)
    for i in range(m - 1):
        perf_1 = np.array(df_perf.loc[df_perf['classifier_name'] == classifiers[i]]['accuracy'])
        for j in range(i + 1, m):
            perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifiers[j]]['accuracy'])
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
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
    rank_data = np.array([df_perf.loc[df_perf['classifier_name'] == c]['accuracy'] for c in classifiers])
    df_ranks = pd.DataFrame(rank_data, index=classifiers).rank(ascending=False)
    average_ranks = df_ranks.mean(axis=1).sort_values(ascending=False)

    return p_values, average_ranks, max_nb_datasets

