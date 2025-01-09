# EV Charging Analysis and Prediction

This repository provides tools and models for analyzing and predicting electric vehicle (EV) charging behavior. It includes data processing, feature engineering, and predictive modeling techniques.

## Key Components

### Jupyter Notebooks
- **01_data_exploration.ipynb**: Data cleaning and exploratory analysis.
- **02_feature_engineering.ipynb**: Feature extraction and transformation.
- **03_xgboost.ipynb**: XGBoost regression model training and evaluation.
- **04_dnn_training.ipynb**: Deep Neural Network (DNN) model training.
- **05_lstm_training.ipynb**: LSTM model for sequence prediction.
- **06_learning_curve_visualization.ipynb**: Learning curve visualization.

### Python Scripts in scr directory
- **data_processing.py**: Data preprocessing and splitting utilities.
- **plot_results.py**: Visualization tools for regression results.
- **UserEmbeddingModel.py**: A DNN incorporating user embeddings.
- **UserEmbeddingLstm.py**: LSTM model with user embeddings.
- **EVChargingModel.py**: Custom model for predicting plugin duration and SOC.

### Features
- Predictive models for EV charging metrics.
- User embeddings for personalization.
- Visualization of results and learning curves.

## Installation

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Preprocess data using `02_feature_engineering.ipynb`.
2. Train models using notebooks like `03_xgboost.ipynb` or `05_lstm_training.ipynb`.
3. Evaluate and visualize results with `06_learning_curve_visualization.ipynb`.


## Download the data by running
    aws s3 cp s3://q658166-thesis/ProcessedData/processed_data.parquet ./processed_data.parquet
    aws s3 cp s3://q658166-thesis/ProcessedData/removed_outliers.parquet ./processed_data.parquet
    
## Best models
    Energy need with DNN: logs/dnn-soc/version_107,   model: checkpoints-dnn/best_checkpoint_soc.ckpt
    Plugin duration with DNN: logs/dnn-dur/version_8, model: checkpoints-dnn/best_checkpoint_dur-v2.ckpt
    Energy need with LSTM: logs/lstm-soc/version_41,  model: checkpoints-lstm/best_checkpoint_soc-v38.ckpt
    Plugin duration with LSTM: logs/lstm-dur/version_0,  model: checkpoints-lstm/best_checkpoint_dur.ckpt