import os
import pandas as pd

class DatasetSplitter:
    def __init__(self, input_file, output_dir="../results/datasets", file_type="detailed_tuning", chunk_size=10000):
        """
        Initializes the DatasetSplitter with the input file path, output directory, file type, and chunk size.

        :param input_file: Path to the CSV file (e.g., detailed tuning results or evaluation metrics).
        :param output_dir: Directory to save the smaller CSV files.
        :param file_type: Type of file being processed ("detailed_tuning" or "evaluation_metrics").
        :param chunk_size: Number of rows per chunk when splitting unstructured large files.
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.file_type = file_type
        self.chunk_size = chunk_size
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data_in_chunks(self):
        """
        Loads the input CSV file in chunks.

        :return: Iterator of DataFrame chunks.
        """
        try:
            print(f"Loading data from {self.input_file} in chunks...")
            return pd.read_csv(self.input_file, chunksize=self.chunk_size, on_bad_lines="skip")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def split_large_file(self):
        """
        Splits a large, unstructured CSV file into smaller chunks.

        Saves each chunk into a separate file.
        """
        print("Splitting large file into smaller chunks...")
        chunk_iterator = self.load_data_in_chunks()
        sub_dir = os.path.join(self.output_dir, f"{self.file_type}_chunks")
        os.makedirs(sub_dir, exist_ok=True)

        for i, chunk in enumerate(chunk_iterator):
            output_file = os.path.join(sub_dir, f"{self.file_type}_chunk_{i + 1}.csv")
            chunk.to_csv(output_file, index=False)
            print(f"Saved chunk {i + 1} to {output_file}")

    def split_by_dataset(self, df):
        """
        Splits the DataFrame into smaller CSV files by dataset.

        :param df: DataFrame containing the data.
        """
        if 'dataset' not in df.columns:
            print("The input file does not contain a 'dataset' column. Cannot proceed.")
            return

        datasets = df['dataset'].unique()
        print(f"Found {len(datasets)} unique datasets: {datasets}")

        for dataset in datasets:
            print(f"Processing dataset: {dataset}")
            dataset_df = df[df['dataset'] == dataset]
            if dataset_df.empty:
                print(f"No data found for dataset {dataset}. Skipping.")
                continue

            # Determine the appropriate subdirectory based on file type
            sub_dir = os.path.join(self.output_dir, f"{self.file_type}_datasets")
            os.makedirs(sub_dir, exist_ok=True)

            # Create a CSV file for the dataset
            output_file = os.path.join(sub_dir, f"{dataset}_{self.file_type}_results.csv")
            dataset_df.to_csv(output_file, index=False)
            print(f"Saved {self.file_type} results to {output_file}")

    def run(self):
        """
        Executes the splitting process based on the file structure.
        """
        try:
            # Check if the file is structured and contains a 'dataset' column
            df_sample = pd.read_csv(self.input_file, nrows=10, on_bad_lines="skip")
            if 'dataset' in df_sample.columns:
                print("File contains 'dataset' column. Proceeding with dataset-based splitting...")
                df = pd.read_csv(self.input_file, on_bad_lines="skip")
                self.split_by_dataset(df)
            else:
                print("File is unstructured. Proceeding with chunk-based splitting...")
                self.split_large_file()
        except Exception as e:
            print(f"Error during processing: {e}")


# Usage Examples
if __name__ == "__main__":
    # For detailed tuning results
    tuning_file = "../results/detailed_tuning_results.csv"
    tuning_splitter = DatasetSplitter(input_file=tuning_file, output_dir="../results/datasets", file_type="detailed_tuning")
    tuning_splitter.run()

    # For evaluation metrics
    evaluation_file = "../results/evaluation_metrics.csv"
    evaluation_splitter = DatasetSplitter(input_file=evaluation_file, output_dir="../results/datasets", file_type="evaluation_metrics", chunk_size=5000)
    evaluation_splitter.run()
