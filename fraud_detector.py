"""
Simple Fraud Detection Model
A beginner-friendly ML project using supervised learning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report


def load_data(file_path):
    """
    Load the fraud dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(df, target_column='is_fraud'):
    """
    Clean and prepare the data for training.
    
    Steps:
    1. Separate features (X) and target (y)
    2. Handle missing values
    3. Scale numeric features
    
    Args:
        df: Raw dataframe
        target_column: Name of the target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Check if target column exists
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return None, None, None, None
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Keep only numeric columns (simple approach for beginners)
    X = X.select_dtypes(include=[np.number])
    
    print(f"\nUsing {X.shape[1]} numeric features")
    print(f"Fraud cases: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"Non-fraud cases: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
    
    # Handle missing values - fill with column mean
    X = X.fillna(X.mean())
    
    # Split into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (important for many ML algorithms)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Random Forest classifier.
    
    Random Forest is a good starting point:
    - Handles non-linear relationships
    - Less prone to overfitting than single decision trees
    - Works well out of the box
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("\nTraining Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print metrics.
    
    Metrics explained:
    - Accuracy: Overall correctness
    - Precision: Of predicted frauds, how many were actually fraud
    - Recall: Of actual frauds, how many did we catch
    - Confusion Matrix: Breakdown of predictions
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\nAccuracy:  {accuracy:.3f} - Overall correctness")
    print(f"Precision: {precision:.3f} - Of predicted frauds, how many were real")
    print(f"Recall:    {recall:.3f} - Of actual frauds, how many we caught")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 No    Yes")
    print(f"Actual No     {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"Actual Yes    {cm[1][0]:5d} {cm[1][1]:5d}")
    
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))
    print("="*60)


def main():
    """
    Main pipeline: load → preprocess → train → evaluate
    """
    # Get dataset path from user
    file_path = input("Enter path to your fraud dataset (CSV): ").strip()
    
    # Load data
    df = load_data(file_path)
    if df is None:
        return
    
    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)
    if X_train is None:
        return
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
