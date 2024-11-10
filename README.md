# Machine Learning Hyperparameter Tuning Analysis

This repository contains the code and documentation for a research project focused on analyzing the tunability of machine learning algorithms. The study investigates various hyperparameter tuning methods applied to different models on multiple datasets, assessing each method's effectiveness, stability, and computational efficiency.

## Project Overview

This project explores the tunability of selected machine learning models using three hyperparameter optimization methods:
1. **Grid Search**
2. **Random Search**
3. **Bayesian Optimization**

The primary objectives are:
- To assess the number of iterations needed for stable optimization results for each tuning method.
- To identify optimal hyperparameter ranges for each model.
- To measure the tunability of individual algorithms and specific hyperparameters.
- To analyze the impact of each hyperparameter tuning technique on model performance and stability.

## Datasets and Models

### Datasets
The study utilizes ten datasets:
- `breast_cancer`
- `iris`
- `california_housing`
- `wine`
- `digits`
- `diabetes`
- `linnerud`
- `auto_mpg`
- `auto_insurance`
- `blood_transfusion`

Each dataset is preprocessed through standardization to ensure uniform scaling of features, enabling consistent performance across different models.

### Models
The following machine learning models were chosen for this study:
- **Random Forest (RandomForestClassifier)**: Tunable hyperparameters include `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- **XGBoost (XGBClassifier)**: Tunable hyperparameters include `n_estimators`, `learning_rate`, `max_depth`, and `subsample`.
- **Elastic Net (ElasticNet)**: Tunable hyperparameters include `alpha`, `l1_ratio`, and `max_iter`.
- **Gradient Boosting (GradientBoostingClassifier)**: Tunable hyperparameters include `n_estimators`, `learning_rate`, and `max_depth`.

## Hyperparameter Tuning Methods

1. **Grid Search**: Exhaustively searches through the specified hyperparameter grid, which ensures the best combination is found but is computationally expensive.
2. **Random Search**: Randomly samples a subset of hyperparameter combinations, allowing a quicker search at the potential expense of finding the absolute optimum.
3. **Bayesian Optimization**: Uses probabilistic models to predict promising areas of the hyperparameter space, improving efficiency over random search while maintaining high accuracy.

Each method employs 3-fold cross-validation to enhance result stability and minimize overfitting risk.

## Experimental Procedure

The experiment was conducted in Python, utilizing libraries like `scikit-learn` and `scikit-optimize`. For each model-dataset combination, the following steps were executed:

1. **Data Loading and Preprocessing**: Each dataset is loaded and standardized.
2. **Task Type Detection**: The experiment checks if the dataset is suitable for the selected model (e.g., classification vs. regression).
3. **Hyperparameter Optimization**: Each tuning method is applied with 15 iterations:
   - For each method, the model's performance (e.g., accuracy or AUC), best hyperparameters, processing time, and memory usage are recorded.
4. **Result Logging**: Results from each iteration are saved in `detailed_tuning_results.csv`. The best scores and parameters for each method are summarized in `best_tuning_results.csv`.
5. **Plot Generation**: Graphs depicting optimization performance over iterations are generated and saved in the `assets/` directory.

## Results and Analysis

The results of the experiments are documented and visualized through multiple figures, saved in the `assets/` directory.

### Key Figures
- **Tunability Scores**: Displays the tunability scores for each tuning method across datasets, indicating the variance in model performance due to hyperparameter adjustments.
- **Score Distributions**: Shows the distribution of model scores for each tuning method, highlighting performance variability.
- **Model Comparison**: A bar plot comparing the best scores achieved for each model on different datasets.
- **Best Model Performance Heatmap**: Heatmap representing the highest scores achieved by each model across tuning methods and datasets.
- **Mean Convergence**: Convergence plots displaying mean model scores over iterations for each tuning method.
- **Critical Difference (CD) Diagram**: Statistical diagram illustrating significant performance differences between tuning methods across models.

### Example Figures
Each figure is stored in the `assets/` directory and can be referenced as follows:
- `assets/tunability_scores_swarmplot.png`
- `assets/score_distributions_violinplot.png`
- `assets/model_comparison_barplot.png`
- `assets/best_model_performance_heatmap.png`
- `assets/mean_convergence_across_datasets.png`
- `assets/cd-diagram.png`

### Summary of Findings
- **Bayesian Optimization** is more efficient in achieving optimal results quickly, especially for models with a large hyperparameter space (e.g., XGBoost).
- Models such as **XGBoost** and **Gradient Boosting** showed higher tunability, improving performance significantly with optimized hyperparameters, while **Elastic Net** exhibited lower tunability.
- **Grid Search** consistently produced stable results but required more computational resources compared to **Random Search** and **Bayesian Optimization**.

## Code Structure

- **`main.py`**: Script to execute the full experiment, including data loading, model selection, hyperparameter tuning, result recording, and plot generation.
- **`hyperparameter_tuning.py`**: Defines functions for each tuning method (`grid_search`, `random_search`, `bayesian_optimization`).
- **`models.py`**: Contains model configurations and hyperparameter ranges for each model used in the experiment.
- **`assets/`**: Directory to store all generated plots and visualizations.

## Dependencies

- Python 3.7+
- `scikit-learn`
- `scikit-optimize`
- `pandas`
- `matplotlib`
- `numpy`

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/repo_name.git
   cd repo_name
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main experiment**:
   ```bash
   python main.py
   ```

## Results Directory

The results are stored in the `results/` directory, including:
- `detailed_tuning_results.csv`: Contains iteration-level details such as scores, parameters, and resource usage.
- `best_tuning_results.csv`: Summary of the best scores and parameters for each model-dataset-method combination.

## License

This project is licensed under the MIT License.

## References

The concept of tunability in machine learning was adapted from:
- Probst, P., et al., "Tunability: Importance of Hyperparameters of Machine Learning Algorithms," 2019.

## Contact

For any inquiries, please contact **@SaintAngeLs**.

