import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Assuming you have a function to load or access the results
def load_tuning_results(results_dir='./results'):
    """Load the tuning results from files or other sources."""
    # Simulated data: In practice, load actual results from your files or experiments.
    data = {
        'dataset': ['breast_cancer', 'breast_cancer', 'breast_cancer', 'iris', 'iris', 'iris'],
        'model': ['elastic_net', 'random_forest', 'xgboost', 'elastic_net', 'random_forest', 'xgboost'],
        'tuning_method': ['grid_search', 'random_search', 'bayes_optimization', 'grid_search', 'random_search', 'bayes_optimization'],
        'score': [0.88, 0.87, 0.89, 0.92, 0.91, 0.93]
    }
    df = pd.DataFrame(data)
    return df

# Load your results into a DataFrame
df_results = load_tuning_results()

# Use ANOVA to compare the means of the three tuning methods across different datasets and models
def perform_anova(df):
    """Performs ANOVA test on tuning results."""
    print("\nPerforming ANOVA test...")
    # Group data by tuning method
    grid_scores = df[df['tuning_method'] == 'grid_search']['score']
    random_scores = df[df['tuning_method'] == 'random_search']['score']
    bayes_scores = df[df['tuning_method'] == 'bayes_optimization']['score']
    
    # ANOVA test
    f_stat, p_value = stats.f_oneway(grid_scores, random_scores, bayes_scores)
    
    print(f"ANOVA results: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
    
    # If p-value < 0.05, we reject the null hypothesis
    if p_value < 0.05:
        print("There is a significant difference between the tuning methods.")
        # Conduct post-hoc analysis (e.g., Tukey's HSD)
        perform_posthoc(df)
    else:
        print("No significant difference found between the tuning methods.")
    
def perform_posthoc(df):
    """Performs Tukey's HSD post-hoc test if ANOVA finds significance."""
    print("\nPerforming Tukey's HSD post-hoc test...")
    tukey = pairwise_tukeyhsd(endog=df['score'], groups=df['tuning_method'], alpha=0.05)
    print(tukey)
    tukey.plot_simultaneous()
    plt.title("Tukey's HSD Test")
    plt.show()

# Visualize the data for clarity
def visualize_results(df):
    """Create a boxplot to visualize score distribution across tuning methods."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='tuning_method', y='score', data=df)
    plt.title("Tuning Method Performance Comparison")
    plt.xlabel("Tuning Method")
    plt.ylabel("Score")
    plt.show()

# Perform the ANOVA test
perform_anova(df_results)

# Visualize the results
visualize_results(df_results)
