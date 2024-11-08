import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
import os
import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import StandardScaler
import tracemalloc  # To track memory usage
import traceback

from data_loader import load_dataset
from models import get_model_and_params
from hyperparameter_tuning import grid_search, random_search, bayesian_optimization
from analysis import plot_results

# List of datasets and models
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
models = ['xgboost', 'random_forest', 'elastic_net', 'gradient_boosting']

# Directory for storing results
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Timeout limit (in seconds)
TIMEOUT = 60 * 20  

# Helper function to standardize the data
def preprocess_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Function to detect task type and avoid incorrect model usage
def is_classification_task(y):
    return type_of_target(y) in ['binary', 'multiclass']

# Function to handle timeouts
def run_with_timeout(func, *args, timeout=TIMEOUT):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            result = future.result(timeout=timeout)
            return result
        except TimeoutError:
            print(f"Timeout occurred in {func.__name__}. Skipping...")
            return None
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            print(traceback.format_exc())  # Log the traceback for debugging
            return None

# Function to store detailed tuning results with time and memory data
def store_iteration_results(dataset_name, model_name, tuning_method, iteration, score, params, wall_time, processor_time, memory_usage):
    scores_path = os.path.join(RESULTS_DIR, 'detailed_tuning_results.csv')
    scores_df = pd.DataFrame({
        'dataset': [dataset_name],
        'model': [model_name],
        'tuning_method': [tuning_method],
        'iteration': [iteration],
        'score': [score],
        'parameters': [str(params)],
        'wall_time': [wall_time],
        'processor_time': [processor_time],
        'memory_usage': [memory_usage]
    })
    if os.path.exists(scores_path):
        scores_df.to_csv(scores_path, mode='a', header=False, index=False)
    else:
        scores_df.to_csv(scores_path, mode='w', header=True, index=False)

# Function to store the best score and parameters for each tuning method
def store_best_score(dataset_name, model_name, tuning_method, best_score, best_params):
    if best_score is not None:
        summary_path = os.path.join(RESULTS_DIR, 'best_tuning_results.csv')
        summary_entry = pd.DataFrame({
            'dataset': [dataset_name],
            'model': [model_name],
            'tuning_method': [tuning_method],
            'best_score': [best_score],
            'best_parameters': [str(best_params)]
        })
        if os.path.exists(summary_path):
            summary_entry.to_csv(summary_path, mode='a', header=False, index=False)
        else:
            summary_entry.to_csv(summary_path, mode='w', header=True, index=False)
    else:
        print(f"No best score to store for {dataset_name} with model {model_name} using {tuning_method}")

# Padding helper function
def pad_history(history):
    """Ensure all lists in the history dictionary have the same length."""
    max_length = max(len(v) for v in history.values())
    for key in history:
        history[key] += [None] * (max_length - len(history[key]))  # Pad with None

# Main experiment function
def run_experiment(dataset_name, model_name):
    try:
        print(f"\nRunning experiments on dataset: {dataset_name} with model: {model_name}")
        
        # Load and preprocess the dataset
        try:
            X, y = load_dataset(dataset_name)
            X = preprocess_data(X)  # Standardizing the features
        except ValueError as e:
            print(f"Dataset loading failed for {dataset_name}: {e}")
            return

        # Check if the model is appropriate for the dataset
        if not is_classification_task(y) and model_name in ['svc', 'gradient_boosting', 'random_forest']:
            print(f"Skipping {model_name} on dataset {dataset_name} due to continuous target.")
            return
        
        # Get model and hyperparameter configuration
        model, params = get_model_and_params(model_name)
        
        # Data structure to hold all tuning history for plotting
        history = {
            'iteration': [],
            'grid_score': [],
            'random_score': [],
            'bayes_score': []
        }
        
        # Execute each tuning method with error handling
        for tuning_method, search_func, score_key in [
                ('grid_search', grid_search, 'grid_score'),
                ('random_search', random_search, 'random_score'),
                ('bayesian_optimization', bayesian_optimization, 'bayes_score')]:
            
            print(f"Running {tuning_method} on {model_name} with dataset {dataset_name}...")
            try:
                for iteration in range(1, 15):  # Using 15 iterations for each tuning method
                    tracemalloc.start()  # Start tracking memory
                    start_wall_time = time.perf_counter()
                    start_processor_time = time.process_time()

                    result = run_with_timeout(search_func, model, params, X, y)
                    
                    wall_time = time.perf_counter() - start_wall_time
                    processor_time = time.process_time() - start_processor_time
                    _, peak_memory = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    if result:
                        best_params, best_score = result

                        # Store the score, time, and memory usage for the current tuning method and iteration
                        store_iteration_results(
                            dataset_name, model_name, tuning_method, iteration, best_score, best_params,
                            wall_time, processor_time, peak_memory
                        )
                        
                        # Store only the best score for summary if it's the last iteration
                        if iteration == 15:
                            store_best_score(dataset_name, model_name, tuning_method, best_score, best_params)
                        
                        # Update history for plotting
                        history['iteration'].append(iteration)
                        history[score_key].append(best_score)
                        
                        print(f"Iteration {iteration} ({tuning_method}): Best params: {best_params}, Score: {best_score}, Wall Time: {wall_time:.2f}s, Processor Time: {processor_time:.2f}s, Memory: {peak_memory / 1024:.2f} KB")
                    else:
                        # Append None for missing results in case of timeout or error
                        history['iteration'].append(iteration)
                        history[score_key].append(None)
            
            except Exception as e:
                print(f"Error during {tuning_method} on {model_name} with dataset {dataset_name}: {e}")
                print(traceback.format_exc())
                # Ensure padding for failed iterations
                for iteration in range(1, 6):
                    history['iteration'].append(iteration)
                    history[score_key].append(None)
        
        # Pad the history lists to ensure they are of the same length
        pad_history(history)
        
        # Plot results for this experiment, checking for any empty data in history
        if any(history[key] for key in ['grid_score', 'random_score', 'bayes_score']):
            plot_title = f"Tuning Results: {dataset_name} - {model_name}"
            plot_filename = f"{dataset_name}_{model_name}_tuning_results.png"
            plot_results(history, title=plot_title, filename=plot_filename)
        else:
            print(f"No scores to plot for {dataset_name} with model {model_name}")
        
    except Exception as e:
        print(f"Failed to run tuning on {model_name} and dataset {dataset_name}: {e}")
        print(traceback.format_exc())

# Parallel experiment execution
for dataset_name in datasets:
    with ProcessPoolExecutor(max_workers=5) as executor:
        future_to_experiment = {executor.submit(run_experiment, dataset_name, model_name): (dataset_name, model_name)
                                for model_name in models}
        
        # Wait for completion and handle exceptions
        for future in as_completed(future_to_experiment):
            dataset_name, model_name = future_to_experiment[future]
            try:
                future.result()  # Raises any exception caught during execution
            except Exception as exc:
                print(f"{dataset_name} with {model_name} generated an exception: {exc}")

print("\nAll experiments completed.")
