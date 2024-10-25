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

def load_dataset(name):
    """
    Loads the dataset specified by name using the appropriate loader.
    
    :param name: Name of the dataset.
    :return: Features (X) and target (y).
    """
    loader = DatasetLoaderFactory.get_loader(name)
    return loader.load()
