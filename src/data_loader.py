import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    fetch_california_housing,
    load_wine,
    load_digits,
    load_diabetes,
    load_linnerud
)
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DatasetLoader:
    def load(self):
        raise NotImplementedError("Subclasses should implement this method.")

class SklearnDatasetLoader(DatasetLoader):
    def __init__(self, sklearn_loader, as_frame=True):
        self.sklearn_loader = sklearn_loader
        self.as_frame = as_frame
    
    def load(self):
        data = self.sklearn_loader(as_frame=self.as_frame)
        X = data.data
        y = data.target
        return X, y

class CSVFileLoader(DatasetLoader):
    def __init__(self, file_path, delimiter=',', header='infer', column_names=None):
        """
        CSV Loader with options for delimiter and column headers.
        
        :param file_path: The path to the CSV file.
        :param delimiter: The delimiter used in the file (default: ',').
        :param header: If the CSV has headers or not (default: 'infer'). Set to None if no headers.
        :param column_names: List of column names if headers are not present.
        """
        self.file_path = file_path
        self.delimiter = delimiter
        self.header = header
        self.column_names = column_names
    
    def load(self):
        data = pd.read_csv(self.file_path, delimiter=self.delimiter, header=self.header, names=self.column_names)
        return data.iloc[:, :-1], data.iloc[:, -1]  # Assuming the last column is the target

class DatasetLoaderFactory:
    @staticmethod
    def get_loader(name):
        # Handle Sklearn datasets
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
        
        # Handle custom CSV datasets with specific requirements
        elif name == "auto_mpg":
            # Note: auto_mpg uses semicolons as delimiters
            return CSVFileLoader('data/auto_1/auto_mpg.csv', delimiter=';', header=0)
        
        elif name == "auto_insurance":
            # Note: auto_insurance lacks headers, so provide the column names
            return CSVFileLoader(
                'data/auto_insurance/auto-insurance.csv', 
                delimiter=',', 
                header=None, 
                column_names=['X', 'Y']
            )
        
        elif name == "blood_transfusion":
            # blood_transfusion uses a comma delimiter and has headers
            return CSVFileLoader('data/blood_transfusion/transfusion.csv', delimiter=',', header=0)
        
        else:
            raise ValueError(f"Dataset '{name}' is not supported.")

def preprocess_targets(y):
    """
    Ensure the target `y` is consistent for classification or regression tasks.
    Converts targets to numerical values if necessary.
    """
    y = column_or_1d(y)  # Ensures y is a 1-dimensional array
    if y.dtype == 'object' or type(y[0]).__name__ == 'str':
        # Convert string labels or mixed types to integers for classification
        y = LabelEncoder().fit_transform(y)
    elif y.dtype in ['float', 'int']:
        # Ensure numeric type for regression tasks
        y = pd.to_numeric(y, errors='coerce')
    return y

def preprocess_features(X):
    """
    Standardize the features using StandardScaler for better model performance.
    
    :param X: Feature matrix
    :return: Scaled feature matrix
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def load_dataset(name):
    """
    Loads the dataset specified by name using the appropriate loader
    and applies preprocessing to the features and targets.
    
    :param name: Name of the dataset.
    :return: Preprocessed features (X) and target (y).
    """
    loader = DatasetLoaderFactory.get_loader(name)
    X, y = loader.load()
    
    # Preprocess features and targets
    X = preprocess_features(X)
    y = preprocess_targets(y)
    
    return X, y
