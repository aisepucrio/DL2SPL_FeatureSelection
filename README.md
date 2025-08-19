# Feature Selection for Deep Learning Models in Software Performance Prediction

## Features

- **Feature Selection**: Implementation of various feature selection methods (Decision Tree, Random Forest, Gradient Boosting, XGBoost, Word2Vec) to identify relevant features for Deep Learning model training.
- **Deep Learning Model Training**: Utilizes PyTorch and PyTorch Lightning for building and training neural network models.
- **Hyperparameter Optimization**: Employs Ray Tune with OptunaSearch and ASHAScheduler for optimizing model hyperparameters, balancing performance and training time.
- **Results Analysis**: Jupyter notebooks for analyzing training time, evaluating model performance (MAPE), and generating plots.

## 1. Methodology

### 1.1 Data Preparation

The dataset is `data (2).parquet`, containing software performance metrics. The target variable is `perf`. `data_features_w2v.json` contains Word2Vec-derived features.

### 1.2 Feature Selection Techniques

- **Decision Tree (DT)**: Feature importance based on impurity reduction.
- **Random Forest (RF)**: Ensemble-averaged impurity reduction.
- **Gradient Boosting (GB)**: Feature contribution to the boosting process.
- **XGBoost (XGB)**: Optimized gradient boosting, importance based on weight, gain, and cover.
- **Word2Vec (W2V)**: Selection of discriminative dimensions from dense vector representations of textual/categorical data.

Feature importance scores are in `feature_importances_DecisionTree.csv`, `feature_importances_RandomForest.csv`, `feature_importances_GradientBoosting.csv`, and `feature_importances_XGBoost.csv`.

### 1.3 Deep Learning Model Training

Deep neural networks are implemented using PyTorch and PyTorch Lightning. Model architecture is defined in `pytorch.py` and `pytorch2.py`, including `LightningModel` for neural network structure, optimizers (Adam, AdamW), loss functions (MSE, SmoothL1Loss, MAE), and activation functions (ReLU, PReLU, ELU).

### 1.4 Hyperparameter Optimization

Ray Tune, with `OptunaSearch` and `ASHAScheduler`, is used for hyperparameter tuning (optimizer, loss function, activation function). Implementations are in `pytorch.py` and `pytorch2.py`.

### 1.5 Performance Evaluation

Model performance is evaluated using Mean Absolute Percentage Error (MAPE) and training time. `Training time.ipynb` analyzes training durations. `generate plots.ipynb` visualizes results. `mape_summary.csv` and `training_times_summary.csv` store aggregated metrics.

## 2. Repository Structure

```
DL2SPL_FeatureSelection/
├── data (2).parquet               # Main dataset for software performance prediction
├── data_features_w2v.json         # Word2Vec features derived from textual/categorical data
├── feature_importances_DecisionTree.csv  # Feature importance scores from Decision Tree
├── feature_importances_GradientBoosting.csv # Feature importance scores from Gradient Boosting
├── feature_importances_RandomForest.csv   # Feature importance scores from Random Forest
├── feature_importances_XGBoost.csv      # Feature importance scores from XGBoost
├── grid_search.py                 # Example script for grid search (hyperparameter tuning)
├── grid_search-2.py               # Another example script for grid search
├── LICENSE                        # MIT License file
├── mape_summary.csv               # Summary of Mean Absolute Percentage Error (MAPE) results
├── plots/                         # Directory for generated plots and visualizations
├── pytorch.py                     # Main script for PyTorch Lightning model training and Ray Tune optimization
├── pytorch2.py                    # Alternative/updated script for PyTorch Lightning model training and Ray Tune optimization
├── pytorch_new_02.ipynb           # Jupyter notebook for new PyTorch experiments (potentially)
├── pytorch_new_2_02.ipynb         # Another Jupyter notebook for new PyTorch experiments
├── README.md                      # Original README file
├── requirements.txt               # Python dependencies for the project
├── results.ipynb                  # Jupyter notebook for analyzing experimental results
├── teste.py                       # Test script (likely for development/debugging)
├── Training time.ipynb            # Jupyter notebook for analyzing model training times
├── training_times_summary.csv     # Summary of model training times
├── clean results.ipynb            # Jupyter notebook for cleaning and organizing experiment results
└── generate plots.ipynb           # Jupyter notebook for generating plots and visualizations
```

## 3. Setup and Usage

### 3.1 Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aisepucrio/DL2SPL_FeatureSelection.git
   cd DL2SPL_FeatureSelection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` specifies `lightning`, `scikit-learn`, `matplotlib`, and `seaborn`.

### 3.2 Running Experiments

- **Model Training and Hyperparameter Optimization:**
  Execute `pytorch.py` or `pytorch2.py` directly:
  ```bash
  python pytorch.py
  # or
  python pytorch2.py
  ```
  (Configured for `cuda` if available, else `cpu`.)

- **Grid Search (Alternative Hyperparameter Tuning):**
  Execute `grid_search.py` or `grid_search-2.py`:
  ```bash
  python grid_search.py
  ```

- **Analyzing Results with Jupyter Notebooks:**
  Start a Jupyter server in the root directory:
  ```bash
  jupyter notebook
  ```
  Navigate to:
    - `Training time.ipynb`: For training duration analysis.
    - `clean results.ipynb`: For cleaning experimental output.
    - `generate plots.ipynb`: For generating plots and visualizations.
    - `results.ipynb`: For general experimental outcomes analysis.

## 4. Results and Discussion

Experimental results are in `mape_summary.csv` and `training_times_summary.csv`. The `plots/` directory contains visual representations. Feature selection reduces training time and can improve predictive accuracy. Detailed discussion is in `generate plots.ipynb` and `results.ipynb`.
