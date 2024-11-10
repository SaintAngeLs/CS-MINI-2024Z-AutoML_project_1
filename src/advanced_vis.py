import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
import numpy as np
import scipy.interpolate as interp

# Load the detailed tuning results
detailed_results = pd.read_csv("../results/detailed_tuning_results.csv", on_bad_lines='skip')

# Select the last iteration for each (dataset, model, tuning_method) as the best result
best_results = detailed_results.sort_values(by=['dataset', 'model', 'tuning_method', 'iteration']) \
    .groupby(['dataset', 'model', 'tuning_method'], as_index=False).last()

# Save the best tuning results to a new file
best_results.to_csv("../results/best_tuning_results.csv", index=False)
print("Best tuning results saved to '../results/best_tuning_results.csv'")

# Pivot the data for heatmap
pivot_data = best_results.pivot_table(
    values='score',
    index=['dataset', 'model'],
    columns='tuning_method',
    aggfunc='max'
)

# Heatmap for best model performance across datasets and tuning methods
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_data, annot=True, cmap='viridis')
plt.title("Best Model Performance across Datasets and Tuning Methods")
plt.ylabel("Dataset - Model")
plt.xlabel("Tuning Method")
plt.xticks(rotation=45)
plt.savefig("../results/best_model_performance_heatmap.png")
plt.close()

# Violin plot for score distributions across tuning methods
plt.figure(figsize=(10, 6))
sns.violinplot(data=detailed_results, x='tuning_method', y='score', inner='point')
plt.title("Score Distributions across Tuning Methods")
plt.xlabel("Tuning Method")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.savefig("../results/score_distributions_violinplot.png")
plt.close()

# Filter to include only the top scores per dataset and model, with deprecation warning fix
top_scores = detailed_results.groupby(['dataset', 'model'], as_index=False).apply(
    lambda df: df.nlargest(1, 'score')
).reset_index(drop=True)

# Scatter matrix for hyperparameter analysis, if columns are available
hyperparameters = ['score', 'alpha', 'l1_ratio', 'max_iter']
available_columns = [col for col in hyperparameters if col in top_scores.columns]

if available_columns:
    fig = px.scatter_matrix(
        top_scores,
        dimensions=available_columns,
        color='dataset',
        symbol='model',
        title="Scatter Plot Matrix of Hyperparameters for Top Scores"
    )
    fig.update_traces(diagonal_visible=False)
    fig.write_image("../results/scatter_matrix_hyperparameters.png")
else:
    print("Required columns for scatter matrix are not available in the data.")

# Parallel coordinates plot for high scores if hyperparameter columns are present
if set(['alpha', 'l1_ratio', 'max_iter']).issubset(detailed_results.columns):
    fig = px.parallel_coordinates(
        detailed_results[detailed_results['score'] > 0.9],  # Filter for high scores
        dimensions=['score', 'alpha', 'l1_ratio', 'max_iter'],
        color='score',
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Parallel Coordinates of Hyperparameters for High Scores"
    )
    fig.write_image("../results/parallel_coordinates_hyperparameters.png")
else:
    print("Required columns for parallel coordinates plot are not available in the data.")

# Bar plot for model comparison by dataset (best scores)
plt.figure(figsize=(12, 6))
sns.barplot(data=best_results, x='dataset', y='score', hue='model')
plt.title("Model Comparison by Dataset (Best Scores)")
plt.xlabel("Dataset")
plt.ylabel("Best Score")
plt.legend(title="Model")
plt.xticks(rotation=45)
plt.savefig("../results/model_comparison_barplot.png")
plt.close()

# Calculate the mean convergence for each dataset-model combination across all tuning methods
mean_convergence = detailed_results.groupby(['dataset', 'tuning_method', 'iteration'], as_index=False)['score'].mean()

# Define datasets for subplots
titles = mean_convergence['dataset'].unique()
num_datasets = len(titles)
cols = 2
rows = math.ceil(num_datasets / cols)

# Create a subplot for each dataset
fig, axs = plt.subplots(rows, cols, figsize=(14, rows * 4))
axs = axs.flatten()
fig.suptitle("Mean Convergence of Hyperparameter Tuning Methods Across Datasets")

# Plot each dataset's convergence for different tuning methods
for i, dataset in enumerate(titles):
    dataset_data = mean_convergence[mean_convergence['dataset'] == dataset]

    # Loop over tuning methods and plot each one
    for method in ['bayesian_optimization', 'grid_search', 'random_search']:
        method_data = dataset_data[dataset_data['tuning_method'] == method].sort_values(by='iteration')
        
        # Plot mean score vs. iterations
        x = method_data['iteration']
        y = method_data['score']
        axs[i].plot(x, y, label=method.replace('_', ' ').capitalize())

    # Setting labels and title for each subplot
    axs[i].set_title(dataset.replace('_', ' ').capitalize())
    axs[i].set_xlabel("Iterations")
    axs[i].set_ylabel("Mean Score (e.g., ROC AUC)")
    axs[i].legend()

# Hide any unused subplots
for j in range(i + 1, len(axs)):
    axs[j].axis('off')

# Save the plot showing mean convergence for each dataset with tuning methods as separate lines
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("mean_convergence_across_datasets.png")
plt.close()

# Line plot for hyperparameter tuning trajectories, if 'parameters' column is available
if 'parameters' in detailed_results.columns:
    fig = px.line(
        detailed_results,
        x='parameters',
        y='score',
        color='tuning_method',
        line_group='model',
        hover_name='dataset',
        title="Hyperparameter Tuning Trajectories"
    )
    fig.update_layout(xaxis_title="Parameter Set", yaxis_title="Score")
    fig.write_image("../results/hyperparameter_tuning_trajectories.png")
else:
    print("Column 'parameters' is not available in detailed_results for the tuning trajectories plot.")


# --- Tunability Plot ---
# Calculate tunability as the standard deviation of scores across iterations for each tuning method and dataset
tunability = (
    detailed_results.groupby(['dataset', 'tuning_method'])['score']
    .std()  # Standard deviation of scores as tunability metric
    .reset_index()
    .rename(columns={'score': 'tunability'})
)

# Replace underscores in tuning method names for better readability
tunability['tuning_method'] = tunability['tuning_method'].str.replace('_', ' ').str.capitalize()

# Plot tunability scores as a swarm plot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='tuning_method', y='tunability', hue='dataset', data=tunability, palette='viridis', size=10)

# Customize plot
plt.title('Tunability Scores for Different Hyperparameter Search Methods')
plt.xlabel('Hyperparameter Search Method')
plt.ylabel('Tunability (Standard Deviation of Score)')
plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the tunability plot
plt.savefig("../results/tunability_scores_swarmplot.png")
plt.close()