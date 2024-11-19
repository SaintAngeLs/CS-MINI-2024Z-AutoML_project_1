import os
import pandas as pd
import matplotlib.pyplot as plt
from critical import draw_cd_diagram

class CriticalDiagram:
    def __init__(self, input_path, output_dir="./preprocessed_data/critical_diagram", alpha=0.05):
        """
        Initializes the CriticalDiagram class with input and output paths.

        :param input_path: Path to the input CSV file containing performance data.
        :param output_dir: Directory to save processed data and diagrams.
        :param alpha: Significance level for statistical tests.
        """
        self.input_path = input_path
        self.output_dir = output_dir
        self.alpha = alpha
        self.prepared_data_path = os.path.join(self.output_dir, "prepared_data_for_cd_diagram.csv")
        self.diagram_path = os.path.join(self.output_dir, "cd-diagram.png")
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_data(self):
        """
        Prepares data for the critical diagram by combining model and tuning method
        into a classifier name and extracting the last iteration for each dataset-classifier pair.
        """
        # Load input data
        df = pd.read_csv(self.input_path)

        # Combine model and tuning method to create a classifier name
        df['classifier_name'] = df['model'] + '_' + df['tuning_method']

        # Sort and keep only the last iteration for each dataset-classifier pair
        df = df.sort_values(by=['dataset', 'classifier_name', 'iteration'])
        df_last_iteration = df.groupby(['classifier_name', 'dataset']).last().reset_index()

        # Prepare the final dataset for the critical diagram
        df_prepared = (
            df_last_iteration[['classifier_name', 'dataset', 'mean_test_score']]
            .rename(columns={'dataset': 'dataset_name', 'mean_test_score': 'accuracy'})
        )

        # Save prepared data
        df_prepared.to_csv(self.prepared_data_path, index=False)
        print(f"Prepared data saved for critical diagram in {self.prepared_data_path}")

    def generate_diagram(self):
        """
        Generates the critical difference diagram using the prepared data and saves it without displaying.
        """
        # Load the prepared data
        df_prepared = pd.read_csv(self.prepared_data_path)

        # Generate the critical difference diagram
        draw_cd_diagram(df_perf=df_prepared, alpha=self.alpha, title="Accuracy", labels=True)
        plt.savefig(self.diagram_path, bbox_inches="tight")
        plt.close()
        print(f"Critical diagram saved as {self.diagram_path}")

    def run(self):
        """
        Runs the entire pipeline: prepare data and generate the critical difference diagram.
        """
        print("Preparing data for critical diagram...")
        self.prepare_data()
        print("Generating critical difference diagram...")
        self.generate_diagram()



if __name__ == "__main__":
    input_path = "./preprocessed_data/all_iterations_unique_tuning.csv"

    cd_generator = CriticalDiagram(input_path=input_path)

    cd_generator.run()
