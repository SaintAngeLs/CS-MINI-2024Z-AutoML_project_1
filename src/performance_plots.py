import os
import pandas as pd
import matplotlib.pyplot as plt


class ResultsVisualizer:
    def __init__(self, tuning_file, output_dir="./plots"):
        """
        Initializes the ResultsVisualizer with input file paths and output directory.

        :param tuning_file: Path to the tuning results CSV file.
        :param output_dir: Directory to save the generated plots.
        """
        self.tuning_file = tuning_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self, file_path):
        """
        Loads the data from a CSV file.

        :param file_path: Path to the input CSV file.
        :return: DataFrame containing the input data.
        """
        return pd.read_csv(file_path, on_bad_lines="skip")

    def plot_performance_scores(self, scores_df):
        """
        Creates performance score plots for each dataset-model combination.

        :param scores_df: DataFrame containing the processed tuning results.
        """
        datasets = scores_df["dataset"].unique()

        for dataset in datasets:
            models = scores_df[scores_df["dataset"] == dataset]["model"].unique()

            for model in models:
                plt.figure(figsize=(10, 6))

                # Filter data for the current dataset-model combination
                df = scores_df[(scores_df["dataset"] == dataset) & (scores_df["model"] == model)].sort_values("iteration")

                # Plot mean test scores for each tuning method
                for method in df["tuning_method"].unique():
                    method_df = df[df["tuning_method"] == method]
                    plt.plot(
                        method_df["iteration"],
                        method_df["mean_test_score"],
                        marker="o",
                        linestyle="--",
                        label=f"{method} - Mean Test Score",
                    )

                plt.title(f"Performance Scores for {dataset} - {model}")
                plt.xlabel("Iteration")
                plt.ylabel("Mean Test Score")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                output_path = os.path.join(self.output_dir, f"{dataset}_{model}_performance_scores.png")
                plt.savefig(output_path)
                plt.close()
                print(f"Saved performance scores plot for {dataset} - {model} at {output_path}")

    def generate_plots(self):
        """
        Generates performance score plots for all dataset-model combinations.
        """
        # Load the tuning data
        tuning_df = self.load_data(self.tuning_file)

        # Generate plots
        self.plot_performance_scores(tuning_df)



if __name__ == "__main__":
    visualizer = ResultsVisualizer(
        tuning_file="./preprocessed_data/all_iterations_unique_tuning.csv",
        output_dir="./performance_plots"
    )
    visualizer.generate_plots()
