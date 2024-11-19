from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils.multiclass import type_of_target

class HyperparameterTuner(BaseEstimator):
    def __init__(self, model, params, cv=3, n_jobs=-1, verbose=0, return_train_score=False):
        self.model = model
        self.params = params
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.return_train_score = return_train_score
        self.scoring = None  # Scoring set dynamically based on target type

    def set_scoring(self, y):
        if type_of_target(y) in ['binary', 'multiclass']:
            self.scoring = 'accuracy'
        else:
            self.scoring = 'neg_mean_squared_error'
            
    def grid_search(self, X, y):
        self.set_scoring(y)
        grid_search = GridSearchCV(self.model, param_grid=self.params, cv=self.cv,
                                   n_jobs=self.n_jobs, scoring=self.scoring, verbose=self.verbose,
                                   return_train_score=self.return_train_score)
        grid_search.fit(X, y)
        return grid_search.best_params_, grid_search.best_score_, grid_search.cv_results_

    def random_search(self, X, y, n_iter=10):
        self.set_scoring(y)
        random_search = RandomizedSearchCV(self.model, param_distributions=self.params,
                                           n_iter=n_iter, cv=self.cv, n_jobs=self.n_jobs,
                                           scoring=self.scoring, verbose=self.verbose,
                                           return_train_score=self.return_train_score)
        random_search.fit(X, y)
        return random_search.best_params_, random_search.best_score_, random_search.cv_results_

    def bayesian_optimization(self, X, y, n_iter=10):
        self.set_scoring(y)
        bayes_search = BayesSearchCV(self.model, search_spaces=self.params, n_iter=n_iter,
                                     cv=self.cv, n_jobs=self.n_jobs, scoring=self.scoring,
                                     verbose=self.verbose)
        bayes_search.fit(X, y)
        return bayes_search.best_params_, bayes_search.best_score_, bayes_search.cv_results_
