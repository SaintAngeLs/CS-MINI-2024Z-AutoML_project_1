from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier
from sklearn.svm import SVC

def get_model_and_params(model_name):
    if model_name == "random_forest":
        model = RandomForestClassifier()
        params = {
            "n_estimators": [10, 50, 100, 200],
            "max_depth": [5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    elif model_name == "xgboost":
        model = XGBClassifier()
        params = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "subsample": [0.6, 0.8, 1.0]
        }
    elif model_name == "elastic_net":
        model = ElasticNet()
        params = {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.1, 0.5, 0.9],
            "max_iter": [1000, 2000, 5000]
        }
    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier()
        params = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7]
        }
    elif model_name == "svc":
        model = SVC()
        params = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        }
    else:
        raise ValueError("Model not supported")
    
    return model, params
