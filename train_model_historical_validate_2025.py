#!/usr/bin/env python3
"""
Train ML Model on Historical S&P 500 Data (2020-2024)
Validate on 2025 YTD Data

Strategy: Train on past years, validate on current year (out-of-sample)
"""

import pandas as pd
from datetime import datetime
import json
import os
import sys

# Import the trading system components
from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLModelTrainer

print("=" * 80)
print("ML MODEL TRAINING - Historical Data (2020-2024)")
print("VALIDATION - 2025 YTD Data (Out-of-Sample)")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/6] Loading S&P 500 Data...")

# Load full historical data
full_data_file = 'sp500_historical_data.csv'
if not os.path.exists(full_data_file):
    print(f"Error: {full_data_file} not found!")
    sys.exit(1)

df_full = pd.read_csv(full_data_file, index_col='Date', parse_dates=True)
df_full = df_full.sort_index()

print(f"✓ Loaded full dataset: {len(df_full)} rows ({df_full.index[0]} to {df_full.index[-1]})")

# ============================================================================
# STEP 2: Split Data - Train (2020-2024) & Validation (2025 YTD)
# ============================================================================
print("\n[2/6] Splitting Data: Train (2020-2024) + Validation (2025)...")

# Training: 2020-2024 (5 years of data)
train_df = df_full[(df_full.index >= '2020-01-01') & (df_full.index < '2025-01-01')].copy()

# Validation: 2025 YTD
val_df = df_full[df_full.index >= '2025-01-01'].copy()

print("\n✓ Data Split:")
print(f"  Training (2020-2024):   {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
print(f"  Validation (2025 YTD):  {len(val_df)} rows ({val_df.index[0]} to {val_df.index[-1]})")
print(f"  Ratio: {len(train_df) / (len(train_df) + len(val_df)) * 100:.1f}% / "
      f"{len(val_df) / (len(train_df) + len(val_df)) * 100:.1f}%")

# ============================================================================
# STEP 3: Feature Engineering
# ============================================================================
print("\n[3/6] Generating Features...")

# Generate features for training data
print("  Generating training features...")
train_features = FeatureEngineering.build_complete_feature_set(train_df, include_volume=True)

# Generate features for validation data
print("  Generating validation features...")
val_features = FeatureEngineering.build_complete_feature_set(val_df, include_volume=True)

# Remove rows with NaN values (from indicator calculation)
train_features_clean = train_features.dropna()
val_features_clean = val_features.dropna()

print("\n✓ Features Generated:")
print(f"  Training features:   {train_features_clean.shape} (after dropna)")
print(f"  Validation features: {val_features_clean.shape} (after dropna)")
print(f"  Total features: {train_features_clean.shape[1]}")

print("\nSample features (first 20):")
for i, feat in enumerate(train_features_clean.columns[:20], 1):
    print(f"  {i:2d}. {feat}")
print("  ... and more")

# ============================================================================
# STEP 4: Create Target Variable
# ============================================================================
print("\n[4/6] Creating Target Variable (Future Returns)...")


def create_target(df, periods=1, threshold=0.0):
    """
    Create binary target: 1 if price goes up, 0 if down

    Parameters:
    -----------
    df : DataFrame with 'close' column
    periods : int, number of periods ahead to predict
    threshold : float, minimum return % to be considered positive

    Returns:
    --------
    Series with binary target
    """
    future_return = df['close'].pct_change(periods).shift(-periods)
    target = (future_return > threshold).astype(int)
    return target


# Create target: predict if price will be higher 1 day ahead
train_target = create_target(train_features_clean, periods=1, threshold=0.0)
val_target = create_target(val_features_clean, periods=1, threshold=0.0)

# Drop rows where target is NaN (last few rows)
mask_train = train_target.notna()
train_features_final = train_features_clean[mask_train]
train_target_final = train_target[mask_train]

mask_val = val_target.notna()
val_features_final = val_features_clean[mask_val]
val_target_final = val_target[mask_val]

print("✓ Target Variable Created:")
print(f"  Training:   {train_target_final.shape}")
print(f"  Validation: {val_target_final.shape}")

print("\nTarget Distribution:")
print("  Training Set:")
print(f"    Up (1):   {train_target_final.sum()} ({train_target_final.mean() * 100:.1f}%)")
print(f"    Down (0): {(~train_target_final.astype(bool)).sum()} ({(1 - train_target_final.mean()) * 100:.1f}%)")
print("  Validation Set:")
print(f"    Up (1):   {val_target_final.sum()} ({val_target_final.mean() * 100:.1f}%)")
print(f"    Down (0): {(~val_target_final.astype(bool)).sum()} ({(1 - val_target_final.mean()) * 100:.1f}%)")

# ============================================================================
# STEP 5: Prepare Features (Remove OHLCV columns)
# ============================================================================
print("\n[5/6] Preparing Features for ML Training...")

# Remove OHLCV columns
exclude_cols = ['open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in train_features_final.columns if col not in exclude_cols]

X_train = train_features_final[feature_cols]
y_train = train_target_final
X_val = val_features_final[feature_cols]
y_val = val_target_final

print("✓ Final Feature Sets:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  y_val:   {y_val.shape}")
print(f"  Features used: {len(feature_cols)}")

# ============================================================================
# STEP 6: Train ML Models
# ============================================================================
print("\n[6/6] Training ML Models...")

# Create output directory
os.makedirs('models', exist_ok=True)

print("\n" + "=" * 80)
print("Training Multiple Models")
print("=" * 80)

models_to_train = {
    'xgboost': 'XGBoost Classifier',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'logistic': 'Logistic Regression'
}

results = {}

for model_type, model_name in models_to_train.items():
    print(f"\n{'=' * 80}")
    print(f"Training: {model_name}")
    print(f"{'=' * 80}")

    try:
        # Initialize trainer
        trainer = MLModelTrainer(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            feature_names=feature_cols
        )

        # Train model
        print(f"\n  Training {model_name} on 2020-2024 data...")
        trainer.train(model_type=model_type)

        # Evaluate on training set
        print("\n  Evaluating on training set...")
        train_metrics = trainer.evaluate(X_train, y_train, dataset_name='Training')

        # Evaluate on validation set (2025 YTD - out of sample)
        print("\n  Evaluating on validation set (2025 YTD)...")
        val_metrics = trainer.evaluate(X_val, y_val, dataset_name='Validation (2025)')

        # Get feature importance
        importance = trainer.get_feature_importance(top_n=20)

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'models/{model_type}_train2020-2024_val2025_{timestamp}.pkl'
        trainer.save_model(model_filename)

        results[model_type] = {
            'name': model_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': importance,
            'model_file': model_filename
        }

        print(f"\n✓ {model_name} Training Complete")
        print(f"  Training Accuracy:   {train_metrics['accuracy']:.3f}")
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"  Validation Precision: {val_metrics['precision']:.3f}")
        print(f"  Validation Recall: {val_metrics['recall']:.3f}")
        print(f"  Validation F1: {val_metrics['f1']:.3f}")
        print(f"  Model saved: {model_filename}")

        if importance is not None:
            print("\n  Top 15 Most Important Features:")
            for i, (feat, imp) in enumerate(importance[:15], 1):
                print(f"    {i:2d}. {feat:35s} {imp:.4f}")

    except Exception as e:
        print(f"\n✗ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()
        results[model_type] = {'error': str(e)}

# ============================================================================
# Model Comparison
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison_data = []
for model_type, result in results.items():
    if 'val_metrics' in result:
        comparison_data.append({
            'Model': result['name'],
            'Train Acc': f"{result['train_metrics']['accuracy']:.3f}",
            'Val Acc': f"{result['val_metrics']['accuracy']:.3f}",
            'Val Precision': f"{result['val_metrics']['precision']:.3f}",
            'Val Recall': f"{result['val_metrics']['recall']:.3f}",
            'Val F1': f"{result['val_metrics']['f1']:.3f}",
            'Val ROC AUC': f"{result['val_metrics'].get('roc_auc', 0):.3f}"
        })

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # Save comparison
    comparison_file = 'models/model_comparison_2020-2024_train_2025_val.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✓ Comparison saved to: {comparison_file}")

# Save training summary
summary = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_period': '2020-2024',
    'val_period': '2025 YTD',
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'num_features': len(feature_cols),
    'models_trained': results
}

summary_file = 'models/training_summary_2020-2024_train_2025_val.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"✓ Training summary saved to: {summary_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

print("\n📊 Training Strategy: Historical Training + Recent Validation")
print(f"  Training: 2020-2024 ({len(X_train)} samples)")
print(f"  Validation: 2025 YTD ({len(X_val)} samples) - OUT OF SAMPLE")

successful_models = [r for r in results.values() if 'val_metrics' in r]
print(f"\n🎯 Models Trained: {len(successful_models)}")
for model_type, result in results.items():
    if 'val_metrics' in result:
        val_acc = result['val_metrics']['accuracy']
        train_acc = result['train_metrics']['accuracy']
        overfitting = train_acc - val_acc
        print(f"  {result['name']:25s} Val: {val_acc:.1%}  Train: {train_acc:.1%}  (Gap: {overfitting:+.1%})")

if successful_models:
    # Find best model
    best_model = max(successful_models, key=lambda x: x['val_metrics']['accuracy'])
    print(f"\n🏆 Best Model: {best_model['name']}")
    print(f"  Validation Accuracy: {best_model['val_metrics']['accuracy']:.1%}")
    print(f"  Model File: {best_model['model_file']}")

print("\n📁 Output Files:")
print("  Models: models/*_train2020-2024_val2025_*.pkl")
print(f"  Comparison: {comparison_file}")
print(f"  Summary: {summary_file}")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Review model comparison - select best performer on 2025 data")
print("  2. Use best model in backtesting with MLStrategy")
print("  3. Test on unseen data or paper trade")
print("=" * 80)
