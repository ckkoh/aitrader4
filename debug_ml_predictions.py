"""
Debug script to check ML model predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path

from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLModelTrainer

# Load data
print("Loading data...")
df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
print(f"Loaded {len(df)} days of data")

# Generate features (with volume=True to match training, standard set only)
print("\nGenerating features...")
df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)
# NO custom S&P 500 features - using standard set only for consistency
df_features = df_features.dropna()

print(f"Total features: {len(df_features.columns)}")
print(f"Feature columns: {df_features.columns.tolist()[:10]}...")  # First 10

# Load a trained model
model_path = 'walkforward_2022_jan_apr/models/Jan_2022.pkl'
print(f"\nLoading model: {model_path}")
trainer = MLModelTrainer.load_model(str(model_path))

# Get test data (Jan 2022)
test_data = df_features.loc['2022-01-01':'2022-01-31']
print(f"\nTest data: {len(test_data)} days")

# Get feature columns used during training
exclude_cols = ['target_class', 'target_regression', 'target_binary',
                'future_return', 'open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in df_features.columns if col not in exclude_cols]

print(f"\nFeature columns for prediction: {len(feature_cols)}")

# Make predictions
print("\nMaking predictions...")
X_test = test_data[feature_cols].values

predictions = trainer.predict(X_test)
probabilities = trainer.predict_proba(X_test)

# Analyze predictions
print(f"\nPredictions summary:")
print(f"  Total predictions: {len(predictions)}")
print(f"  Buy signals (1): {(predictions == 1).sum()}")
print(f"  No-buy signals (0): {(predictions == 0).sum()}")

# Check confidences
max_confidences = probabilities.max(axis=1)
print(f"\nConfidence analysis:")
print(f"  Min confidence: {max_confidences.min():.3f}")
print(f"  Max confidence: {max_confidences.max():.3f}")
print(f"  Mean confidence: {max_confidences.mean():.3f}")
print(f"  Median confidence: {np.median(max_confidences):.3f}")

# Check how many meet threshold
for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
    count = (max_confidences >= threshold).sum()
    print(f"  Confidence >= {threshold:.2f}: {count}/{len(max_confidences)} ({count/len(max_confidences)*100:.1f}%)")

# Show daily predictions
print(f"\nDaily predictions (first 10):")
print(f"{'Date':<12} {'Prediction':<12} {'Confidence':<12}")
print("-" * 40)
for i in range(min(10, len(test_data))):
    date = test_data.index[i].date()
    pred = predictions[i]
    conf = max_confidences[i]
    print(f"{date} {pred:>11} {conf:>11.3f}")
