from data_loader import load_dataset
from models import get_model_and_params
from hyperparameter_tuning import grid_search, random_search, bayesian_optimization
from analysis import plot_results

# Load data
X, y = load_dataset('breast_cancer')

# Choose model and parameters
model_name = 'xgboost'  # You can also try 'lightgbm' or 'catboost'
model, params = get_model_and_params(model_name)

# Run tuning methods
print("Running Grid Search...")
best_params_grid, best_score_grid = grid_search(model, params, X, y)
print(f"Best parameters (Grid Search): {best_params_grid}, Best score: {best_score_grid}")

print("Running Random Search...")
best_params_random, best_score_random = random_search(model, params, X, y)
print(f"Best parameters (Random Search): {best_params_random}, Best score: {best_score_random}")

print("Running Bayesian Optimization...")
best_params_bayes, best_score_bayes = bayesian_optimization(model, params, X, y)
print(f"Best parameters (Bayesian Optimization): {best_params_bayes}, Best score: {best_score_bayes}")

# Example of logging tuning history and visualizing results across iterations
history = [
    {'iteration': 1, 'grid_score': best_score_grid, 'random_score': best_score_random, 'bayes_score': best_score_bayes},
    {'iteration': 2, 'grid_score': best_score_grid - 0.01, 'random_score': best_score_random - 0.02, 'bayes_score': best_score_bayes + 0.01},
    {'iteration': 3, 'grid_score': best_score_grid + 0.02, 'random_score': best_score_random + 0.01, 'bayes_score': best_score_bayes + 0.03}
]

# Visualize the results
plot_results(history, title="Comparison of Grid Search, Random Search, and Bayesian Optimization")
