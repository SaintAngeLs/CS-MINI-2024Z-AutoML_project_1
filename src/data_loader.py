import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, fetch_california_housing

def load_dataset(name):
    if name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        return data.frame, data.target
    elif name == "iris":
        data = load_iris(as_frame=True)
        return data.frame, data.target
    elif name == "california_housing":
        data = fetch_california_housing(as_frame=True)
        return data.frame, data.target
    else:
        # Load custom CSV datasets from 'data/' directory
        return pd.read_csv(f'data/{name}.csv')

