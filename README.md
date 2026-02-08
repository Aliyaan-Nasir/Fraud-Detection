# Fraud Detection Model

A simple machine learning project that detects fraudulent transactions using supervised learning.

## What This Does

Takes a CSV dataset with transaction data and trains a Random Forest classifier to predict fraud vs non-fraud cases.

Outputs:
- Accuracy, precision, and recall metrics
- Confusion matrix
- Detailed classification report

## How It Works

1. **Load data** - Reads CSV file
2. **Preprocess** - Handles missing values, scales features, splits train/test
3. **Train** - Fits a Random Forest model (100 trees)
4. **Evaluate** - Tests on unseen data and prints metrics

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script:
```bash
python fraud_detector.py
```

When prompted, enter the path to your CSV file.

## Dataset Requirements

Your CSV should have:
- Numeric feature columns (transaction amount, time, etc.)
- A target column named `is_fraud` with values 0 (not fraud) or 1 (fraud)

Example structure:
```
amount,time,merchant_id,is_fraud
100.50,14:30,12345,0
5000.00,02:15,67890,1
```

## Sample Dataset

You can test with public datasets like:
- [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)

Just make sure to rename the target column to `is_fraud`.

## Limitations

This is a learning project with real constraints:

- **Only uses numeric features** - Ignores categorical data (could be improved with encoding)
- **Simple preprocessing** - Just fills missing values with mean
- **No hyperparameter tuning** - Uses default Random Forest settings
- **No feature engineering** - Uses raw features as-is
- **Imbalanced data handling** - Doesn't use SMOTE or class weights
- **No model persistence** - Doesn't save the trained model
- **No cross-validation** - Single train/test split only

## Why These Metrics Matter

- **Accuracy** - Good overall measure, but misleading if fraud is rare (e.g., 99% non-fraud means 99% accuracy by always predicting "not fraud")
- **Precision** - Important to avoid false alarms (blocking legitimate transactions)
- **Recall** - Critical for catching actual fraud (missing fraud is costly)

In fraud detection, recall is often more important than precision.

## Future Improvements

If I continue this project, I might add:
- Handle categorical features (one-hot encoding)
- Try other models (XGBoost, Logistic Regression)
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Handle class imbalance (SMOTE, class weights)
- Save and load trained models
- Cross-validation for better evaluation
- ROC curve and AUC score

## Why This Exists

I wanted to:
1. Learn supervised machine learning basics
2. Understand classification metrics
3. Build something practical and simple
4. Practice clean code structure

## License

MIT - Use this however you want.
