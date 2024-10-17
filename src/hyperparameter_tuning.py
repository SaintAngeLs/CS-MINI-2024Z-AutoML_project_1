from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
import numpy as np

def grid_search(model, params, X, y):
    grid_search = GridSearchCV(model, param_grid=params, cv=3, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_score_

def random_search(model, params, X, y, n_iter=10):
    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter, cv=3, n_jobs=-1)
    random_search.fit(X, y)
    return random_search.best_params_, random_search.best_score_

def bayesian_optimization(model, params, X, y, n_iter=10):
    bayes_search = BayesSearchCV(model, search_spaces=params, n_iter=n_iter, cv=3, n_jobs=-1)
    bayes_search.fit(X, y)
    return bayes_search.best_params_, bayes_search.best_score_
