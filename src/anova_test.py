import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Load tuning results from a single detailed CSV file
def load_tuning_results(file='../results/detailed_tuning_results.csv'):
    """Load tuning results from the provided CSV file."""
    return pd.read_csv(file,  on_bad_lines='warn')

# Visualize results with multiple standalone plots
def visualize_results(df, save_dir='plots'):
    """Create and save separate boxplots to compare performance scores across tuning methods and models for each dataset."""
    
    # Ensure the directory for saving plots exists
    os.makedirs(save_dir, exist_ok=True)
    
    unique_datasets = df['dataset'].unique()
    
    for dataset in unique_datasets:
        plt.figure(figsize=(10, 6))
        subset = df[df['dataset'] == dataset]
        
        # Check data distribution with a stripplot
        sns.stripplot(x="model", y="score", hue="tuning_method", data=subset, dodge=True,
                      palette={"grid_search": "blue", "random_search": "orange", "bayesian_optimization": "green"},
                      marker="o", alpha=0.7)
        
        # Add titles and labels
        plt.title(f"Comparison of Tuning Methods for {dataset} Dataset")
        plt.xlabel("Model")
        plt.ylabel("Score")
        
        # Place the legend outside the plot
        plt.legend(title="Tuning Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the plot to the specified directory
        plot_filename = os.path.join(save_dir, f"{dataset}_tuning_methods_comparison.png")
        plt.savefig(plot_filename, bbox_inches='tight')
        
        # Close the plot to free memory
        plt.close()

# Perform ANOVA test
def anova_test(df):
    """Perform ANOVA test across different tuning methods for each dataset and model."""
    unique_datasets = df['dataset'].unique()
    
    anova_results = []
    
    for dataset in unique_datasets:
        subset = df[df['dataset'] == dataset]
        unique_models = subset['model'].unique()
        
        for model in unique_models:
            model_subset = subset[subset['model'] == model]
            
            # Extract score lists for each tuning method
            scores_grid = model_subset[model_subset['tuning_method'] == 'grid_search']['score']
            scores_random = model_subset[model_subset['tuning_method'] == 'random_search']['score']
            scores_bayes = model_subset[model_subset['tuning_method'] == 'bayesian_optimization']['score']
            
            # Perform ANOVA if there are enough samples
            if len(scores_grid) > 1 and len(scores_random) > 1 and len(scores_bayes) > 1:
                f_stat, p_val = f_oneway(scores_grid, scores_random, scores_bayes)
                anova_results.append({
                    'dataset': dataset,
                    'model': model,
                    'F-statistic': f_stat,
                    'p-value': p_val
                })
                print(f"ANOVA for {dataset} - {model}: F = {f_stat:.2f}, p = {p_val:.4f}")
            else:
                print(f"Insufficient data for ANOVA in {dataset} - {model}")

    return pd.DataFrame(anova_results)

# Load results and visualize
df_results = load_tuning_results()
visualize_results(df_results)

# Perform ANOVA test and display the results
anova_results_df = anova_test(df_results)

# Optional: Save ANOVA results to a CSV file
anova_results_df.to_csv('anova_results.csv', index=False)
print("ANOVA results saved to 'anova_results.csv'")
