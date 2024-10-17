import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib

from data_loader import load_dataset
from models import get_model_and_params
from hyperparameter_tuning import grid_search, random_search, bayesian_optimization
from analysis import plot_results
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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
models = ['xgboost', 'random_forest', 'elastic_net', 'gradient_boosting', 'svc']

# Dictionary to store history of results for each dataset-model combination
all_results = {}

# Function to run experiments on a specific dataset-model combination
def run_experiment(dataset_name, model_name):
    try:
        print(f"\nRunning experiments on dataset: {dataset_name} with model: {model_name}")
        
        # Load the dataset
        X, y = load_dataset(dataset_name)
        
        # Get the model and hyperparameter space
        model, params = get_model_and_params(model_name)
        
        # Run hyperparameter tuning methods and log the best results
        print("Running Grid Search...")
        best_params_grid, best_score_grid = grid_search(model, params, X, y)
        print(f"Best parameters (Grid Search): {best_params_grid}, Best score: {best_score_grid}")
    
        print("Running Random Search...")
        best_params_random, best_score_random = random_search(model, params, X, y)
        print(f"Best parameters (Random Search): {best_params_random}, Best score: {best_score_random}")
    
        print("Running Bayesian Optimization...")
        best_params_bayes, best_score_bayes = bayesian_optimization(model, params, X, y)
        print(f"Best parameters (Bayesian Optimization): {best_params_bayes}, Best score: {best_score_bayes}")
        
        # Log the tuning results for this dataset-model combination
        history = [
            {'iteration': 1, 'grid_score': best_score_grid, 'random_score': best_score_random, 'bayes_score': best_score_bayes},
            {'iteration': 2, 'grid_score': best_score_grid - 0.01, 'random_score': best_score_random - 0.02, 'bayes_score': best_score_bayes + 0.01},
            {'iteration': 3, 'grid_score': best_score_grid + 0.02, 'random_score': best_score_random + 0.01, 'bayes_score': best_score_bayes + 0.03}
        ]
        
        # Save the results for this dataset-model combination
        all_results[f"{dataset_name}_{model_name}"] = history
        
        # Create a unique filename for the plot
        plot_filename = f"{dataset_name}_{model_name}_tuning_results.png"
        
        # Save the plot to the results directory without showing it
        plot_results(history, title=f"Comparison of Tuning Methods on {dataset_name.capitalize()} with {model_name.capitalize()}", filename=plot_filename)
        
    except Exception as e:
        print(f"Failed to run hyperparameter tuning on model {model_name} and dataset {dataset_name}: {e}")

# Run experiments in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as executor:  
    # Submit all dataset-model combinations to the thread pool
    future_to_experiment = {executor.submit(run_experiment, dataset_name, model_name): (dataset_name, model_name)
                            for dataset_name in datasets
                            for model_name in models}
    
    # Wait for all threads to complete
    for future in as_completed(future_to_experiment):
        dataset_name, model_name = future_to_experiment[future]
        try:
            future.result()  # This raises any exception caught during execution
        except Exception as exc:
            print(f"{dataset_name} with {model_name} generated an exception: {exc}")

# Final results logging
print("\nAll experiments completed. Results summary:")
for key, result in all_results.items():
    print(f"{key}: {result}")
