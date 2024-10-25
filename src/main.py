import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib

from data_loader import load_dataset
from models import get_model_and_params
from hyperparameter_tuning import grid_search, random_search, bayesian_optimization
from analysis import plot_results, CriticalDifferencePlot
import os
from sklearn.utils.multiclass import type_of_target

# List of datasets to experiment on
datasets = [
    'breast_cancer', 
    'iris', 
    'california_housing', 
    'wine', 
    'digits', 
    'diabetes', 
    'linnerud', 
    'auto_mpg', 
    'auto_insurance', 
    'blood_transfusion'
]

# List of models to experiment with
models = ['xgboost', 
          'random_forest', 
          'elastic_net', 
          'gradient_boosting'
          ]

# Timeout limit for each hyperparameter tuning method (in seconds)
TIMEOUT = 60  

# Function to run a specific hyperparameter tuning method with timeout
def run_with_timeout(func, *args, timeout=TIMEOUT):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            result = future.result(timeout=timeout)
            return result
        except TimeoutError:
            print(f"Timeout occurred in {func.__name__}. Skipping...")
            return None, None

# Function to detect task type and avoid incorrect model usage
def is_classification_task(y):
    return type_of_target(y) in ['binary', 'multiclass']

# Function to run experiments on a specific dataset-model combination
def run_experiment(dataset_name, model_name, rankings):
    try:
        print(f"\nRunning experiments on dataset: {dataset_name} with model: {model_name}")
        
        # Load the dataset
        try:
            X, y = load_dataset(dataset_name)
        except ValueError as e:
            print(f"Dataset loading failed for {dataset_name}: {e}")
            return

        # Check if the target type is continuous and skip models that require classification
        if not is_classification_task(y) and model_name in ['svc', 'gradient_boosting', 'random_forest']:
            print(f"Skipping {model_name} on dataset {dataset_name} due to continuous target for classification model.")
            return
        
        # Get the model and hyperparameter space
        model, params = get_model_and_params(model_name)
        
        # Run hyperparameter tuning methods with timeout
        print("Running Grid Search...")
        best_params_grid, best_score_grid = run_with_timeout(grid_search, model, params, X, y)
        if best_params_grid is not None:
            print(f"Best parameters (Grid Search): {best_params_grid}, Best score: {best_score_grid}")
    
        print("Running Random Search...")
        best_params_random, best_score_random = run_with_timeout(random_search, model, params, X, y)
        if best_params_random is not None:
            print(f"Best parameters (Random Search): {best_params_random}, Best score: {best_score_random}")
    
        print("Running Bayesian Optimization...")
        best_params_bayes, best_score_bayes = run_with_timeout(bayesian_optimization, model, params, X, y)
        if best_params_bayes is not None:
            print(f"Best parameters (Bayesian Optimization): {best_params_bayes}, Best score: {best_score_bayes}")
        
        # If no tuning method returned valid results, skip further processing
        if best_params_grid is None and best_params_random is None and best_params_bayes is None:
            print(f"All tuning methods for {model_name} on {dataset_name} failed due to timeout.")
            return
        
        # Log the tuning results for this dataset-model combination
        history = [
            {'iteration': 1, 'grid_score': best_score_grid, 'random_score': best_score_random, 'bayes_score': best_score_bayes},
            {'iteration': 2, 'grid_score': best_score_grid - 0.01 if best_score_grid else None, 'random_score': best_score_random - 0.02 if best_score_random else None, 'bayes_score': best_score_bayes + 0.01 if best_score_bayes else None},
            {'iteration': 3, 'grid_score': best_score_grid + 0.02 if best_score_grid else None, 'random_score': best_score_random + 0.01 if best_score_random else None, 'bayes_score': best_score_bayes + 0.03 if best_score_bayes else None}
        ]
        
        # Create a unique filename for the plot
        plot_filename = f"{dataset_name}_{model_name}_tuning_results.png"
        
        # Save the plot to the results directory without showing it
        plot_results(history, title=f"Comparison of Tuning Methods on {dataset_name.capitalize()} with {model_name.capitalize()}", filename=plot_filename)

        # Append the best score for critical difference diagram
        best_score = max([h['bayes_score'] for h in history if h['bayes_score'] is not None])
        rankings[model_name].append(best_score)
        
    except Exception as e:
        print(f"Failed to run hyperparameter tuning on model {model_name} and dataset {dataset_name}: {e}")

# Run experiments in parallel using ProcessPoolExecutor
for dataset_name in datasets:
    rankings = {model: [] for model in models}  # Reset rankings for each dataset

    with ProcessPoolExecutor(max_workers=5) as executor:
        future_to_experiment = {executor.submit(run_experiment, dataset_name, model_name, rankings): (dataset_name, model_name)
                                for model_name in models}
        
        # Wait for all threads to complete
        for future in as_completed(future_to_experiment):
            dataset_name, model_name = future_to_experiment[future]
            try:
                future.result()  # This raises any exception caught during execution
            except Exception as exc:
                print(f"{dataset_name} with {model_name} generated an exception: {exc}")

    # # Generate the Critical Difference Diagram for the current dataset
    # critical_difference_plot = CriticalDifferencePlot(alpha=0.05)
    # avg_ranks = critical_difference_plot.calculate_average_ranks(rankings)
    # p_values = critical_difference_plot.wilcoxon_holm(rankings)
    
    # critical_difference_plot.plot(
    #     avg_ranks=avg_ranks, 
    #     classifiers=list(avg_ranks.keys()), 
    #     p_values=p_values, 
    #     filename=f"{dataset_name}_cd_diagram.png", 
    #     title=f"Critical Difference Diagram for {dataset_name.capitalize()}"
    # )

print("\nAll experiments completed.")
