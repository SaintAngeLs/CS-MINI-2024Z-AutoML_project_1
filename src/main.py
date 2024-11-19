import os
import time
import pandas as pd
import tracemalloc
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor, TimeoutError
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score

from data_loader import load_dataset
from models import get_model_and_params
from hyperparameter_tuning import HyperparameterTuner

class ExperimentManager:
    def __init__(self, datasets, models, results_dir='./results', timeout=60 * 20):
        self.datasets = datasets
        self.models = models
        self.results_dir = results_dir
        self.timeout = timeout
        os.makedirs(self.results_dir, exist_ok=True)

    def is_classification_task(self, y):
        """Detects if the task is classification."""
        return type_of_target(y) in ['binary', 'multiclass']

    def calculate_metrics(self, y_true, y_pred, task_type):
        """Calculates evaluation metrics."""
        if task_type == "classification":
            metrics = classification_report(y_true, y_pred, output_dict=True)
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        else:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
        return metrics

    def store_evaluation_metrics(self, dataset_name, model_name, tuning_method, iteration, metrics):
        """Stores evaluation metrics to a CSV file."""
        eval_path = os.path.join(self.results_dir, 'evaluation_metrics.csv')
        metrics_df = pd.DataFrame({
            'dataset': [dataset_name],
            'model': [model_name],
            'tuning_method': [tuning_method],
            'iteration': [iteration],
            **{f"metric_{k}": [v] for k, v in metrics.items()}
        })
        if os.path.exists(eval_path):
            metrics_df.to_csv(eval_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(eval_path, mode='w', header=True, index=False)

    def store_iteration_results(self, dataset_name, model_name, tuning_method, iteration, results, wall_time, processor_time, memory_usage):
        """Stores iteration results with performance metrics."""
        scores_path = os.path.join(self.results_dir, 'detailed_tuning_results.csv')
        scores_df = pd.DataFrame({
            'dataset': [dataset_name] * len(results['mean_test_score']),
            'model': [model_name] * len(results['mean_test_score']),
            'tuning_method': [tuning_method] * len(results['mean_test_score']),
            'iteration': [iteration] * len(results['mean_test_score']),
            'mean_test_score': results['mean_test_score'],
            'std_test_score': results['std_test_score'],
            'parameters': [str(params) for params in results['params']],
            'wall_time': [wall_time] * len(results['mean_test_score']),
            'processor_time': [processor_time] * len(results['mean_test_score']),
            'memory_usage': [memory_usage] * len(results['mean_test_score']),
        })
        if os.path.exists(scores_path):
            scores_df.to_csv(scores_path, mode='a', header=False, index=False)
        else:
            scores_df.to_csv(scores_path, mode='w', header=True, index=False)

    def run_with_timeout(self, func, *args):
        """Executes a function with a specified timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args)
            try:
                return future.result(timeout=self.timeout)
            except TimeoutError:
                print(f"Timeout occurred in {func.__name__}. Skipping execution...")
                return None
            except Exception as e:
                print(f"An error occurred in {func.__name__}: {e}")
                print(traceback.format_exc())
                return None

    def run_experiment(self, dataset_name, model_name):
        """Runs experiments for a single dataset-model pair."""
        try:
            print(f"\nRunning experiments on dataset: {dataset_name} with model: {model_name}")

            # Load and preprocess the dataset
            X, y = load_dataset(dataset_name)
            task_type = "classification" if self.is_classification_task(y) else "regression"

            # Validate model compatibility
            if task_type == "regression" and model_name in ['svc', 'random_forest']:
                print(f"Skipping {model_name} on dataset {dataset_name} due to incompatible task type.")
                return

            # Initialize model and hyperparameter tuner
            model, params = get_model_and_params(model_name)
            tuner = HyperparameterTuner(model, params)

            # Execute tuning methods
            for tuning_method, search_func in [
                    ('grid_search', tuner.grid_search),
                    ('random_search', tuner.random_search),
                    ('bayesian_optimization', tuner.bayesian_optimization)]:

                print(f"Running {tuning_method} on {model_name} with dataset {dataset_name}...")
                for iteration in range(1, 16):  # Iterate through 15 runs
                    tracemalloc.start()
                    start_wall_time = time.perf_counter()
                    start_processor_time = time.process_time()

                    result = self.run_with_timeout(search_func, X, y)

                    wall_time = time.perf_counter() - start_wall_time
                    processor_time = time.process_time() - start_processor_time
                    _, peak_memory = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    if result:
                        best_params, best_score, cv_results = result
                        model.set_params(**best_params)
                        model.fit(X, y)
                        y_pred = model.predict(X)
                        metrics = self.calculate_metrics(y, y_pred, task_type)

                        # Store results
                        self.store_evaluation_metrics(dataset_name, model_name, tuning_method, iteration, metrics)
                        self.store_iteration_results(
                            dataset_name, model_name, tuning_method, iteration,
                            cv_results, wall_time, processor_time, peak_memory
                        )

                        print(f"Iteration {iteration} ({tuning_method}): Best params: {best_params}, "
                              f"Metrics: {metrics}, Wall Time: {wall_time:.2f}s, "
                              f"Processor Time: {processor_time:.2f}s, Memory: {peak_memory / 1024:.2f} KB")
                    else:
                        print(f"Iteration {iteration} ({tuning_method}): Skipped due to timeout or error.")
        except Exception as e:
            print(f"Failed to run tuning on {model_name} and dataset {dataset_name}: {e}")
            print(traceback.format_exc())

    def run_all_experiments(self):
        """Runs experiments in parallel across datasets and models."""
        for dataset_name in self.datasets:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future_to_experiment = {
                    executor.submit(self.run_experiment, dataset_name, model_name): (dataset_name, model_name)
                    for model_name in self.models
                }
                for future in as_completed(future_to_experiment):
                    dataset_name, model_name = future_to_experiment[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"{dataset_name} with {model_name} generated an exception: {exc}")

# Main execution
if __name__ == "__main__":
    datasets = [
        'breast_cancer', 'iris', 'california_housing', 'wine',
        'digits', 'diabetes', 'linnerud', 'auto_mpg',
        'auto_insurance', 'blood_transfusion'
    ]
    models = ['xgboost', 'random_forest', 'elastic_net', 'gradient_boosting']

    experiment_manager = ExperimentManager(datasets, models)
    experiment_manager.run_all_experiments()
