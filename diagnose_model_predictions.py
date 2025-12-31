#!/usr/bin/env python3
"""
Diagnose what the ML model is actually predicting
Check confidence scores and class distribution
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ml_training_pipeline import MLModelTrainer
from feature_engineering import FeatureEngineering

# Load one of the trained models
model_path = 'regime_adaptive_results/model_split_9_adaptive.pkl'  # 2024 Q1 - should be BULL market

print(f"\n{'='*80}")
print(f"MODEL PREDICTION DIAGNOSIS")
print(f"{'='*80}")
print(f"\nModel: {model_path}")

# Load model
import pickle
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

# Extract model from saved data
if isinstance(model_data, dict):
    model = model_data.get('model', model_data)
else:
    model = model_data

# Load data
print("\nLoading S&P 500 data...")
df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)

# Need to include training data for feature calculation, but will focus on test period
train_start = '2022-12-28'  # Split 9 train start
test_start = '2023-12-29'  # Split 9 test start
test_end = '2024-04-01'    # Split 9 test end

df = df.loc[train_start:test_end]

print(f"Data range (train+test): {df.index[0]} to {df.index[-1]}")
print(f"Test period: {test_start} to {test_end}")

# Generate features
print("\nGenerating features...")
df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)
df_features = df_features.dropna()

print(f"Clean data points: {len(df_features)}")

# Load top 20 features
top_features_file = 'feature_selection_results/top_20_features.csv'
top_features_df = pd.read_csv(top_features_file)
top_features = top_features_df['feature'].tolist()

print(f"\nUsing {len(top_features)} features")

# Make predictions
print("\n" + "="*80)
print("PREDICTIONS")
print("="*80)

predictions = []
for idx in range(len(df_features)):
    row = df_features.iloc[idx]
    features = row[top_features].values.reshape(1, -1)

    proba = model.predict_proba(features)[0]
    prob_sell = proba[0]
    prob_buy = proba[1]
    predicted_class = int(prob_buy > 0.5)

    predictions.append({
        'date': df_features.index[idx],
        'prob_sell': prob_sell,
        'prob_buy': prob_buy,
        'predicted_class': predicted_class,
        'predicted_label': 'BUY' if predicted_class == 1 else 'SELL/HOLD'
    })

pred_df = pd.DataFrame(predictions)

# Filter to test period only
pred_df['date'] = pd.to_datetime(pred_df['date'])
pred_df = pred_df[(pred_df['date'] >= test_start) & (pred_df['date'] <= test_end)]

print(f"\nFiltered to test period: {len(pred_df)} days")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nBUY Probability Statistics:")
print(f"  Min:    {pred_df['prob_buy'].min():.4f}")
print(f"  Max:    {pred_df['prob_buy'].max():.4f}")
print(f"  Mean:   {pred_df['prob_buy'].mean():.4f}")
print(f"  Median: {pred_df['prob_buy'].median():.4f}")
print(f"  Std:    {pred_df['prob_buy'].std():.4f}")

print(f"\nSELL/HOLD Probability Statistics:")
print(f"  Min:    {pred_df['prob_sell'].min():.4f}")
print(f"  Max:    {pred_df['prob_sell'].max():.4f}")
print(f"  Mean:   {pred_df['prob_sell'].mean():.4f}")
print(f"  Median: {pred_df['prob_sell'].median():.4f}")
print(f"  Std:    {pred_df['prob_sell'].std():.4f}")

print(f"\nPredicted Class Distribution:")
print(f"  SELL/HOLD (0): {(pred_df['predicted_class'] == 0).sum()} ({(pred_df['predicted_class'] == 0).sum() / len(pred_df) * 100:.1f}%)")
print(f"  BUY (1):       {(pred_df['predicted_class'] == 1).sum()} ({(pred_df['predicted_class'] == 1).sum() / len(pred_df) * 100:.1f}%)")

# Threshold analysis
print(f"\n" + "="*80)
print("THRESHOLD ANALYSIS")
print("="*80)

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
print(f"\n{'Threshold':<12} {'BUY Signals':<15} {'Percentage':<12}")
print("-" * 40)
for thresh in thresholds:
    buy_signals = (pred_df['prob_buy'] >= thresh).sum()
    pct = buy_signals / len(pred_df) * 100
    print(f"{thresh:<12.2f} {buy_signals:<15} {pct:<12.1f}%")

# Show top 10 highest BUY probabilities
print(f"\n" + "="*80)
print("TOP 10 HIGHEST BUY PROBABILITIES")
print("="*80)

top_10 = pred_df.nlargest(10, 'prob_buy')[['date', 'prob_buy', 'prob_sell', 'predicted_label']]
print(top_10.to_string(index=False))

# Show bottom 10 lowest BUY probabilities
print(f"\n" + "="*80)
print("BOTTOM 10 LOWEST BUY PROBABILITIES")
print("="*80)

bottom_10 = pred_df.nsmallest(10, 'prob_buy')[['date', 'prob_buy', 'prob_sell', 'predicted_label']]
print(bottom_10.to_string(index=False))

# Export to CSV for detailed analysis
output_file = 'model_predictions_diagnosis.csv'
pred_df.to_csv(output_file, index=False)
print(f"\n✅ Full predictions saved to: {output_file}")

print(f"\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
