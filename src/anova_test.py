import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


class TuningResultsAnalyzer:
    def __init__(self, tuning_file, output_dir="./analysis_results"):
        """
        Initializes the TuningResultsAnalyzer with the input file path and output directory.

        :param tuning_file: Path to the tuning results CSV file.
        :param output_dir: Directory to save plots and ANOVA results.
        """
        self.tuning_file = tuning_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None

    def load_data(self):
        """
        Loads the tuning results from the provided CSV file.

        :return: DataFrame containing the tuning results.
        """
        self.df = pd.read_csv(self.tuning_file, on_bad_lines="warn")
        return self.df

    def visualize_results(self):
        """
        Creates and saves boxplots and stripplots to compare performance scores across
        tuning methods and models for each dataset.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please call `load_data` first.")

        unique_datasets = self.df["dataset"].unique()

        for dataset in unique_datasets:
            plt.figure(figsize=(10, 6))
            subset = self.df[self.df["dataset"] == dataset]

            # Add boxplot to visualize the distribution of scores for each tuning method and model
            sns.boxplot(
                x="model",
                y="mean_test_score",
                hue="tuning_method",
                data=subset,
                dodge=True,
                fliersize=0,
                palette={
                    "grid_search": "blue",
                    "random_search": "orange",
                    "bayesian_optimization": "green",
                },
            )

            # Overlay stripplot to show individual data points
            sns.stripplot(
                x="model",
                y="mean_test_score",
                hue="tuning_method",
                data=subset,
                dodge=True,
                palette={
                    "grid_search": "blue",
                    "random_search": "orange",
                    "bayesian_optimization": "green",
                },
                marker="o",
                alpha=0.7,
            )

            # Avoid duplicate legends
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(
                handles[:3], labels[:3], title="Tuning Method", bbox_to_anchor=(1.05, 1), loc="upper left"
            )

            # Add titles and labels
            plt.title(f"Comparison of Tuning Methods for {dataset} Dataset")
            plt.xlabel("Model")
            plt.ylabel("Mean Test Score")

            # Save the plot to the specified directory
            plot_filename = os.path.join(
                self.output_dir, f"{dataset}_tuning_methods_comparison.png"
            )
            plt.savefig(plot_filename, bbox_inches="tight")

            # Close the plot to free memory
            plt.close()
            print(f"Saved performance scores plot for {dataset} at {plot_filename}")

    def perform_anova(self):
        """
        Performs ANOVA tests across different tuning methods for each dataset and model.

        :return: DataFrame containing ANOVA results (dataset, model, F-statistic, p-value).
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please call `load_data` first.")

        unique_datasets = self.df["dataset"].unique()
        anova_results = []

        for dataset in unique_datasets:
            subset = self.df[self.df["dataset"] == dataset]
            unique_models = subset["model"].unique()

            for model in unique_models:
                model_subset = subset[subset["model"] == model]

                # Extract score lists for each tuning method
                scores_grid = model_subset[model_subset["tuning_method"] == "grid_search"][
                    "mean_test_score"
                ]
                scores_random = model_subset[model_subset["tuning_method"] == "random_search"][
                    "mean_test_score"
                ]
                scores_bayes = model_subset[model_subset["tuning_method"] == "bayesian_optimization"][
                    "mean_test_score"
                ]

                # Perform ANOVA if there are enough samples
                if len(scores_grid) > 1 and len(scores_random) > 1 and len(scores_bayes) > 1:
                    f_stat, p_val = f_oneway(scores_grid, scores_random, scores_bayes)
                    anova_results.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "F-statistic": f_stat,
                            "p-value": p_val,
                        }
                    )
                    print(
                        f"ANOVA for {dataset} - {model}: F = {f_stat:.2f}, p = {p_val:.4f}"
                    )
                else:
                    print(f"Insufficient data for ANOVA in {dataset} - {model}")

        results_df = pd.DataFrame(anova_results)
        results_file = os.path.join(self.output_dir, "anova_results.csv")
        results_df.to_csv(results_file, index=False)
        print(f"ANOVA results saved to {results_file}")

        return results_df


# Usage Example
if __name__ == "__main__":
    analyzer = TuningResultsAnalyzer(
        tuning_file="./preprocessed_data/all_iterations_unique_tuning.csv",
        output_dir="./anova_analysis_results",
    )

    # Load data
    analyzer.load_data()

    # Visualize results
    analyzer.visualize_results()

    # Perform ANOVA tests and save results
    anova_results_df = analyzer.perform_anova()
