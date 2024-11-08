import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the data, skipping problematic lines
best_results = pd.read_csv("../results/best_tuning_results.csv")
detailed_results = pd.read_csv("../results/detailed_tuning_results.csv", on_bad_lines='skip')

# Pivot the data for heatmap
pivot_data = best_results.pivot_table(
    values='best_score',
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
plt.show()

# Violin plot for score distributions across tuning methods
plt.figure(figsize=(10, 6))
sns.violinplot(data=detailed_results, x='tuning_method', y='score', inner='point')
plt.title("Score Distributions across Tuning Methods")
plt.xlabel("Tuning Method")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()

# Filter to include only the best scores per dataset and model
top_scores = detailed_results.groupby(['dataset', 'model'], as_index=False).apply(
    lambda df: df.nlargest(1, 'score')
).reset_index(drop=True)

# Check if required columns are available for scatter matrix plot
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
    fig.show()
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
    fig.show()
else:
    print("Required columns for parallel coordinates plot are not available in the data.")

# Bar plot for model comparison by dataset (best scores)
plt.figure(figsize=(12, 6))
sns.barplot(data=best_results, x='dataset', y='best_score', hue='model')
plt.title("Model Comparison by Dataset (Best Scores)")
plt.xlabel("Dataset")
plt.ylabel("Best Score")
plt.legend(title="Model")
plt.xticks(rotation=45)
plt.show()

# Line plot for hyperparameter tuning trajectories
if 'parameters' in detailed_results.columns:
    fig = px.line(
        detailed_results,
        x='parameters',  # Assuming parameter changes are sequential
        y='score',
        color='tuning_method',
        line_group='model',
        hover_name='dataset',
        title="Hyperparameter Tuning Trajectories"
    )
    fig.update_layout(xaxis_title="Parameter Set", yaxis_title="Score")
    fig.show()
else:
    print("Column 'parameters' is not available in detailed_results for the tuning trajectories plot.")
