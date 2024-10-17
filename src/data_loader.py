import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, fetch_california_housing, load_wine, load_digits, load_diabetes, load_linnerud

class DatasetLoader:
    def load(self):
        raise NotImplementedError("Subclasses should implement this method.")

class SklearnDatasetLoader(DatasetLoader):
    def __init__(self, sklearn_loader, as_frame=True):
        self.sklearn_loader = sklearn_loader
        self.as_frame = as_frame
    
    def load(self):
        data = self.sklearn_loader(as_frame=self.as_frame)
        return data.frame, data.target

class CSVFileLoader(DatasetLoader):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        data = pd.read_csv(self.file_path)
        return data.iloc[:, :-1], data.iloc[:, -1]  # Assuming last column is the target

class DatasetLoaderFactory:
    @staticmethod
    def get_loader(name):
        if name == "breast_cancer":
            return SklearnDatasetLoader(load_breast_cancer)
        elif name == "iris":
            return SklearnDatasetLoader(load_iris)
        elif name == "california_housing":
            return SklearnDatasetLoader(fetch_california_housing)
        elif name == "wine":
            return SklearnDatasetLoader(load_wine)
        elif name == "digits":
            return SklearnDatasetLoader(load_digits)
        elif name == "diabetes":
            return SklearnDatasetLoader(load_diabetes)
        elif name == "linnerud":
            return SklearnDatasetLoader(load_linnerud)
        elif name == "auto_mpg":
            return CSVFileLoader('data/auto_1/auto_mpg.csv')
        elif name == "auto_insurance":
            return CSVFileLoader('data/auto_insurance/auto-insurance.csv')
        elif name == "blood_transfusion":
            return CSVFileLoader('data/blood_transfusion/transfusion.csv')
        else:
            raise ValueError(f"Dataset '{name}' is not supported.")

def load_dataset(name):
    loader = DatasetLoaderFactory.get_loader(name)
    return loader.load()

# # Example Usage
# if __name__ == "__main__":
#     # Load sklearn dataset
#     X, y = load_dataset("breast_cancer")
#     print(X.head(), y.head())

#     # Load custom CSV dataset
#     X, y = load_dataset("blood_transfusion")
#     print(X.head(), y.head())
