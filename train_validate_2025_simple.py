#!/usr/bin/env python3
"""
Simple ML Training: Train on 2020-2024, Validate on 2025 YTD
Direct sklearn implementation without complex pipelines
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import pickle

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)

# Local imports
from feature_engineering import FeatureEngineering

print("=" * 80)
print("ML TRAINING: 2020-2024 → VALIDATION: 2025 YTD")
print("=" * 80)

# Load data
print("\n[1/5] Loading S&P 500 Data...")
df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
df = df.sort_index()

# Split: Train (2020-2024) + Val (2025)
train_df = df[(df.index >= '2020-01-01') & (df.index < '2025-01-01')]
val_df = df[df.index >= '2025-01-01']

print(f"  Train: {len(train_df)} rows ({train_df.index[0].date()} to {train_df.index[-1].date()})")
print(f"  Val:   {len(val_df)} rows ({val_df.index[0].date()} to {val_df.index[-1].date()})")

# Generate features
print("\n[2/5] Generating Features...")
train_feat = FeatureEngineering.build_complete_feature_set(train_df, include_volume=True).dropna()
val_feat = FeatureEngineering.build_complete_feature_set(val_df, include_volume=True).dropna()

print(f"  Train features: {train_feat.shape}")
print(f"  Val features:   {val_feat.shape}")

# Create target (next day return > 0)
print("\n[3/5] Creating Target...")
train_target = (train_feat['close'].pct_change().shift(-1) > 0).astype(int)[:-1]
val_target = (val_feat['close'].pct_change().shift(-1) > 0).astype(int)[:-1]

train_feat = train_feat.iloc[:-1]
val_feat = val_feat.iloc[:-1]

# Prepare features
exclude = ['open', 'high', 'low', 'close', 'volume']
feature_cols = [c for c in train_feat.columns if c not in exclude]

X_train = train_feat[feature_cols]
y_train = train_target
X_val = val_feat[feature_cols]
y_val = val_target

# Clean data: replace inf with NaN, then fill NaN with 0
print("\n  Cleaning data (replacing inf/NaN)...")
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"  Target dist (train): Up={y_train.sum()}/{len(y_train)} ({y_train.mean() * 100:.1f}%)")
print(f"  Target dist (val):   Up={y_val.sum()}/{len(y_val)} ({y_val.mean() * 100:.1f}%)")

# Train models
print("\n[4/5] Training Models...")

os.makedirs('models', exist_ok=True)

models = {
    'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Evaluate
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred)
    val_rec = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feat_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        feat_imp = None

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file = f'models/{name.lower()}_2020-2024train_2025val_{timestamp}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols}, f)

    results[name] = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_precision': val_prec,
        'val_recall': val_rec,
        'val_f1': val_f1,
        'feature_importance': feat_imp,
        'model_file': model_file
    }

    print(f"    Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}")
    print(f"    Saved: {model_file}")

# Model comparison
print("\n[5/5] Model Comparison...")

comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Acc': [r['train_acc'] for r in results.values()],
    'Val Acc': [r['val_acc'] for r in results.values()],
    'Val Precision': [r['val_precision'] for r in results.values()],
    'Val Recall': [r['val_recall'] for r in results.values()],
    'Val F1': [r['val_f1'] for r in results.values()]
})

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(comparison.to_string(index=False))

# Save comparison
comp_file = 'models/comparison_2020-2024train_2025val.csv'
comparison.to_csv(comp_file, index=False)

# Show feature importance for best model
best_model_name = comparison.loc[comparison['Val Acc'].idxmax(), 'Model']
best_result = results[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"  Validation Accuracy: {best_result['val_acc']:.1%}")
print(f"  Validation F1: {best_result['val_f1']:.3f}")
print(f"  Model: {best_result['model_file']}")

if best_result['feature_importance'] is not None:
    print("\n  Top 15 Features:")
    for i, row in best_result['feature_importance'].head(15).iterrows():
        print(f"    {i + 1:2d}. {row['feature']:30s} {row['importance']:.4f}")

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'train_period': '2020-2024',
    'val_period': '2025 YTD',
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'features': len(feature_cols),
    'best_model': best_model_name,
    'best_val_acc': float(best_result['val_acc']),
    'results': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                    for kk, vv in v.items() if kk != 'feature_importance'}
                for k, v in results.items()}
}

summary_file = 'models/summary_2020-2024train_2025val.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n📁 Saved:")
print(f"  Comparison: {comp_file}")
print(f"  Summary: {summary_file}")
print("  Models: models/*_2020-2024train_2025val_*.pkl")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!")
print("=" * 80)
