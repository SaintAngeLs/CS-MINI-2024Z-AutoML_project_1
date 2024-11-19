import os
import pandas as pd
import ast


class BasePreprocessor:
    def __init__(self, file_path, output_dir="./preprocessed_data"):
        """
        Base class for preprocessing CSV files.

        :param file_path: Path to the input CSV file.
        :param output_dir: Directory to save the preprocessed CSV files.
        """
        self.file_path = file_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        Loads the data from a CSV file.

        :return: DataFrame containing the input data.
        """
        return pd.read_csv(self.file_path, on_bad_lines="skip")

    def deduplicate_per_iteration(self, df):
        """
        Deduplicates data by retaining the last appearance for each unique combination
        of 'dataset', 'model', 'tuning_method', and 'iteration'.

        :param df: DataFrame to deduplicate.
        :return: Deduplicated DataFrame with one row per iteration.
        """
        df = df.sort_values(by=["dataset", "model", "tuning_method", "iteration"]).reset_index(drop=True)
        deduplicated_df = df.groupby(["dataset", "model", "tuning_method", "iteration"], as_index=False).last()
        return deduplicated_df


class TuningResultsPreprocessor(BasePreprocessor):
    def process(self):
        """
        Executes the preprocessing pipeline for tuning results:
        1. Load data
        2. Deduplicate per iteration
        3. Save deduplicated data to an output file
        """
        tuning_df = self.load_data()
        deduplicated_tuning_df = self.deduplicate_per_iteration(tuning_df)
        deduplicated_tuning_df = deduplicated_tuning_df.sort_values(
            by=["dataset", "model", "tuning_method", "iteration"]
        ).reset_index(drop=True)
        output_path = os.path.join(self.output_dir, "all_iterations_unique_tuning.csv")
        deduplicated_tuning_df.to_csv(output_path, index=False)
        print(f"Saved deduplicated tuning results to {output_path}")


class MetricsPreprocessor(BasePreprocessor):
    def parse_metrics(self, metrics_df):
        """
        Parses metrics columns containing dictionary-like strings into separate columns.

        :param metrics_df: DataFrame containing metrics as stringified dictionaries.
        :return: DataFrame with parsed metrics expanded into individual columns.
        """
        metric_columns = [col for col in metrics_df.columns if col.startswith("metric_")]
        for col in metric_columns:
            metrics_df = self.expand_metrics_column(metrics_df, col)
        print(f"Parsed metrics columns: {metric_columns}")
        return metrics_df

    def expand_metrics_column(self, df, column_name):
        """
        Expands a column containing dictionary-like strings into separate columns.

        :param df: DataFrame containing the column to expand.
        :param column_name: Name of the column to expand.
        :return: DataFrame with expanded columns.
        """
        if df[column_name].notna().any():
            def safe_eval(val):
                try:
                    # Safely parse dictionary-like strings
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    return {}

            parsed_metrics = df[column_name].dropna().apply(safe_eval)

            # Normalize the parsed dictionaries into a DataFrame
            metrics_df = pd.json_normalize(parsed_metrics)

            # Prepend the original column name to the new columns
            metrics_df.columns = [f"{column_name}_{subcol}" for subcol in metrics_df.columns]

            # Join the expanded metrics with the original DataFrame
            df = df.join(metrics_df)
        return df

    def process(self):
        """
        Executes the preprocessing pipeline for evaluation metrics:
        1. Load data
        2. Parse metrics data
        3. Deduplicate per iteration
        4. Save deduplicated and sorted data to an output file
        """
        metrics_df = self.load_data()
        parsed_metrics_df = self.parse_metrics(metrics_df)
        deduplicated_metrics_df = self.deduplicate_per_iteration(parsed_metrics_df)

        # Ensure sorting by iteration
        deduplicated_metrics_df = deduplicated_metrics_df.sort_values(
            by=["dataset", "model", "tuning_method", "iteration"]
        ).reset_index(drop=True)

        output_path = os.path.join(self.output_dir, "all_iterations_unique_metrics.csv")
        deduplicated_metrics_df.to_csv(output_path, index=False)
        print(f"Saved deduplicated evaluation metrics to {output_path}")


class CSVPreprocessorCoordinator:
    def __init__(self, tuning_file, metrics_file, output_dir="./preprocessed_data"):
        """
        Coordinates the preprocessing for tuning results and evaluation metrics.

        :param tuning_file: Path to the tuning results CSV file.
        :param metrics_file: Path to the evaluation metrics CSV file.
        :param output_dir: Directory to save the preprocessed CSV files.
        """
        self.tuning_preprocessor = TuningResultsPreprocessor(tuning_file, output_dir)
        self.metrics_preprocessor = MetricsPreprocessor(metrics_file, output_dir)

    def process(self):
        """
        Executes the preprocessing for both tuning results and evaluation metrics.
        """
        self.tuning_preprocessor.process()
        self.metrics_preprocessor.process()



if __name__ == "__main__":
    coordinator = CSVPreprocessorCoordinator(
        tuning_file="../results/detailed_tuning_results.csv",
        metrics_file="../results/evaluation_metrics.csv",
        output_dir="./preprocessed_data"
    )
    coordinator.process()
