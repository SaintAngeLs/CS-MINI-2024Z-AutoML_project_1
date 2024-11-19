# AutoML -- Hyperparameter Tuning Analysis -- Project 

This directory contains the source code for analyzing and visualizing various aspects of the AutoML project. Below is an overview of the structure and description of the plots generated.

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Plot Descriptions](#plot-descriptions)
   - [Evaluation Metrics Plots](#evaluation-metrics-plots)
   - [ANOVA Analysis Results](#anova-analysis-results)
   - [Performance Plots](#performance-plots)
   - [Critical Diagram](#critical-diagram)
   - [Tunability Scores Plot](#tunability-scores-plot)
3. [How to Run](#how-to-run)

---

## Directory Structure

```plaintext
.
├── advanced_vis.py               # Visualization scripts for evaluation metrics and tunability
├── analysis.py                   # Analysis functions
├── anova_analysis_results/       # ANOVA comparison plots and results
├── critical_diagram_main.py      # Script for generating critical diagrams
├── evaluation_metrics_plots/     # Metric distribution and accuracy trend plots for datasets
├── hyperparameter_tuning.py      # Core tuning algorithms
├── performance_plots/            # Performance scores across models and datasets
├── preprocessed_data/            # Preprocessed data for CD diagrams and other analyses
├── tunability_scores_swarmplot/  # Tunability swarm plot
```

---

## Plot Descriptions

### 1. Evaluation Metrics Plots

**Location:** `evaluation_metrics_plots/`

For each dataset, the following plots are generated:
- **Accuracy Trend Plot**: Tracks accuracy across iterations for each model and tuning method.
- **Precision Distribution Plot**: Distribution of precision across models and tuning methods.
- **Recall Distribution Plot**: Distribution of recall across models and tuning methods.
- **F1-Score Distribution Plot**: Distribution of F1-scores across models and tuning methods.

Example:
- `evaluation_metrics_plots/blood_transfusion/accuracy_trend.png`
- `evaluation_metrics_plots/blood_transfusion/precision_distribution.png`

---

### 2. ANOVA Analysis Results

**Location:** `anova_analysis_results/`

Compares the performance of tuning methods across datasets using ANOVA analysis. Each dataset has a tuning method comparison plot.

Example:
- `anova_analysis_results/blood_transfusion_tuning_methods_comparison.png`
- `anova_analysis_results/wine_tuning_methods_comparison.png`

---

### 3. Performance Plots

**Location:** `performance_plots/`

Displays the performance of different models for various datasets. Each plot corresponds to a specific model and dataset.

Example:
- `performance_plots/iris_gradient_boosting_performance_scores.png`
- `performance_plots/wine_xgboost_performance_scores.png`

---

### 4. Critical Diagram

**Location:** `preprocessed_data/critical_diagram/`

- **Critical Diagram Plot**: Visualizes the ranking of different tuning methods across datasets.
- **Prepared Data for CD Diagram**: The processed dataset used for generating critical diagrams.

Example:
- `preprocessed_data/critical_diagram/cd-diagram.png`

---

### 5. Tunability Scores Plot

**Location:** `tunability_scores_swarmplot/`

- **Tunability Plot**: Measures the variability (standard deviation) of scores across iterations for each tuning method and dataset.
- Excludes the `auto_insurance` dataset.

Example:
- `tunability_scores_swarmplot/tunability_scores_swarmplot.png`

---

## How to Run

1. Clone the repository and navigate to the `src` directory.
2. Ensure all required Python libraries are installed (see `requirements.txt`).
3. Run specific scripts for generating plots:
   - **File with the main data for experiment**
    ```bash
    python main.py # will run experiment and create  composite 'detailed_tuning_results.csv' and 'evaluation_metrics.csv' in ''../results/'' directory
    ```
   - **Evaluation Metrics Plots**:
     ```bash
     python advanced_vis.py
     ```
   - **ANOVA Analysis**:
     ```bash
     python anova_test.py
     ```
   - **Critical Diagram**:
     ```bash
     python critical_diagram_main.py
     ```
   - **Tunability Plot**:
     ```bash
     python advanced_vis.py
     ```
4. Check the respective directories for output plots and results.

---

## Dependencies

- Python 3.10+
- Required Libraries:
  - `pandas`
  - `seaborn`
  - `matplotlib`
  - `numpy`
