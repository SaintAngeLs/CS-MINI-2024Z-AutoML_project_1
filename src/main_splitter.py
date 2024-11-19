import os
import pandas as pd

class DatasetSplitter:
    def __init__(self, input_file, output_dir="../results/datasets"):
        """
        Initializes the DatasetSplitter with the input file path and output directory.

        :param input_file: Path to the detailed tuning results CSV file.
        :param output_dir: Directory to save the smaller CSV files for each dataset.
        """
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        Loads the detailed tuning results CSV file.

        :return: DataFrame containing the dataset.
        """
        try:
            print(f"Loading data from {self.input_file}...")
            df = pd.read_csv(self.input_file, on_bad_lines="skip")
            print("Data loaded successfully.")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def split_by_dataset(self, df):
        """
        Splits the DataFrame into smaller CSV files by dataset.

        :param df: DataFrame containing the detailed tuning results.
        """
        datasets = df['dataset'].unique()
        print(f"Found {len(datasets)} unique datasets: {datasets}")

        for dataset in datasets:
            print(f"Processing dataset: {dataset}")
            dataset_df = df[df['dataset'] == dataset]
            if dataset_df.empty:
                print(f"No data found for dataset {dataset}. Skipping.")
                continue

            # Create a CSV file for the dataset
            output_file = os.path.join(self.output_dir, f"{dataset}_results.csv")
            dataset_df.to_csv(output_file, index=False)
            print(f"Saved dataset results to {output_file}")

    def run(self):
        """
        Executes the splitting process.
        """
        df = self.load_data()
        if df.empty:
            print("The input file is empty. Exiting.")
            return

        self.split_by_dataset(df)


if __name__ == "__main__":
    input_file = "../results/detailed_tuning_results.csv"
    splitter = DatasetSplitter(input_file=input_file)
    splitter.run()
