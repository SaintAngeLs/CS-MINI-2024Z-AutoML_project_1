import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast


class MetricsVisualizer:
    def __init__(self, input_path, output_dir="./evaluation_metrics_plots"):
        """
        Initializes the MetricsVisualizer with the input data path and output directory.

        :param input_path: Path to the evaluation metrics CSV file.
        :param output_dir: Directory to save the plots.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        Loads the evaluation metrics data, skipping problematic rows.

        :return: DataFrame containing the full dataset.
        """
        try:
            df = pd.read_csv(self.input_path, on_bad_lines="skip")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def extract_metrics(self, row, metric_column):
        """
        Extracts metric values from a dictionary-like string in the specified column.

        :param row: DataFrame row.
        :param metric_column: Column name containing metric data as a string.
        :return: Parsed dictionary of metrics or an empty dictionary if parsing fails.
        """
        try:
            if isinstance(row[metric_column], str) and "{" in row[metric_column]:
                return ast.literal_eval(row[metric_column])
            else:
                return {}
        except (ValueError, SyntaxError, TypeError):
            return {}

    def plot_metric_distribution(self, df, metric_column, metric_key, dataset):
        """
        Plots the distribution of a specific metric for each model and tuning method for a given dataset.

        :param df: DataFrame containing filtered data for the dataset.
        :param metric_column: Column name containing metric data as a string.
        :param metric_key: Specific metric key to visualize (e.g., 'precision').
        :param dataset: Name of the dataset being processed.
        """
        # Extract the metric key from the metric dictionary
        df.loc[:, metric_key] = df.apply(
            lambda row: self.extract_metrics(row, metric_column).get(metric_key, None), axis=1
        )

        if df[metric_key].isnull().all():
            print(f"Warning: No valid '{metric_key}' data found in '{metric_column}' for dataset '{dataset}'. Skipping.")
            return

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="model", y=metric_key, hue="tuning_method")
        plt.title(f"{metric_key.capitalize()} Distribution by Model and Tuning Method ({dataset})")
        plt.xlabel("Model")
        plt.ylabel(metric_key.capitalize())
        plt.xticks(rotation=45)
        plt.legend(title="Tuning Method")
        dataset_dir = os.path.join(self.output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        plot_path = os.path.join(dataset_dir, f"{metric_key}_distribution.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {metric_key} distribution plot for {dataset} to {plot_path}")

    def plot_accuracy_trend(self, df, dataset):
        """
        Plots accuracy trends over iterations for each model and tuning method for a given dataset.

        :param df: DataFrame containing filtered data for the dataset.
        :param dataset: Name of the dataset being processed.
        """
        if "metric_accuracy" not in df.columns or df["metric_accuracy"].isnull().all():
            print(f"Warning: No valid accuracy data for dataset '{dataset}'. Skipping accuracy trend plot.")
            return

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x="iteration", y="metric_accuracy", hue="model", style="tuning_method", markers=True)
        plt.title(f"Accuracy Trend Over Iterations ({dataset})")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.legend(title="Model - Tuning Method")
        dataset_dir = os.path.join(self.output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        plot_path = os.path.join(dataset_dir, "accuracy_trend.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved accuracy trend plot for {dataset} to {plot_path}")

    def run(self):
        """
        Executes the visualization pipeline for all datasets in the input file.
        """
        print("Loading data...")
        df = self.load_data()
        if df.empty:
            print("No data available. Exiting.")
            return
        print("Data loaded successfully.")

        # Process each dataset separately
        datasets = df['dataset'].unique()
        for dataset in datasets:
            print(f"Processing dataset: {dataset}")
            dataset_df = df[df['dataset'] == dataset].copy()
            if dataset_df.empty:
                print(f"No data available for {dataset}. Skipping.")
                continue

            print("Generating metric distribution plots...")
            for metric_col, metric_key in [
                ("metric_0", "precision"),
                ("metric_1", "recall"),
                ("metric_macro avg", "f1-score"),
            ]:
                self.plot_metric_distribution(dataset_df, metric_col, metric_key, dataset)

            print("Generating accuracy trend plot...")
            self.plot_accuracy_trend(dataset_df, dataset)



if __name__ == "__main__":
    input_path = "../results/evaluation_metrics.csv"
    visualizer = MetricsVisualizer(input_path=input_path)
    visualizer.run()
