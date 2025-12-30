#!/usr/bin/env python3
"""
Train ML Model on 2025 YTD S&P 500 Data
Using the complete feature engineering and ML training pipeline
"""

import pandas as pd
from datetime import datetime
import json
import os
import sys

# Import the trading system components
from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLTradingPipeline, MLModelTrainer

print("=" * 80)
print("ML MODEL TRAINING - 2025 YTD S&P 500 DATA")
print("=" * 80)

# ============================================================================
# STEP 1: Load 2025 YTD Data
# ============================================================================
print("\n[1/6] Loading 2025 YTD S&P 500 Data...")

data_file = 'sp500_ytd_2025.csv'
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found!")
    print("Please run 'python3 download_sp500_data.py' first.")
    sys.exit(1)

df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
df = df.sort_index()

print(f"✓ Loaded {len(df)} rows")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")
print(f"  Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())
print("\nLast few rows:")
print(df.tail())

# ============================================================================
# STEP 2: Split Data - Training (First 80%) & Validation (Last 20%)
# ============================================================================
print("\n[2/6] Splitting Data into Training and Validation Sets...")

# Calculate split point (80/20)
split_idx = int(len(df) * 0.80)

train_df = df.iloc[:split_idx].copy()
val_df = df.iloc[split_idx:].copy()

print("\n✓ Data Split:")
print(f"  Training:   {len(train_df)} rows ({train_df.index[0]} to {train_df.index[-1]})")
print(f"  Validation: {len(val_df)} rows ({val_df.index[0]} to {val_df.index[-1]})")
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
train_features = train_features.dropna()
val_features = val_features.dropna()

print("\n✓ Features Generated:")
print(f"  Training features:   {train_features.shape}")
print(f"  Validation features: {val_features.shape}")
print(f"  Total features: {train_features.shape[1]}")

# Display sample features
print("\nSample features:")
print(train_features.columns.tolist()[:20])
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
train_target = create_target(train_features, periods=1, threshold=0.0)
val_target = create_target(val_features, periods=1, threshold=0.0)

# Drop rows where target is NaN (last few rows)
train_features = train_features[train_target.notna()]
train_target = train_target[train_target.notna()]
val_features = val_features[val_target.notna()]
val_target = val_target[val_target.notna()]

print("✓ Target Variable Created:")
print(f"  Training target: {train_target.shape}")
print(f"  Validation target: {val_target.shape}")
print("\nTarget Distribution (Training):")
print(f"  Up (1):   {train_target.sum()} ({train_target.mean() * 100:.1f}%)")
print(f"  Down (0): {(~train_target.astype(bool)).sum()} ({(1 - train_target.mean()) * 100:.1f}%)")

# ============================================================================
# STEP 5: Prepare Features (Remove OHLCV columns)
# ============================================================================
print("\n[5/6] Preparing Features for ML Training...")

# Remove OHLCV columns and target-related columns
exclude_cols = ['open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in train_features.columns if col not in exclude_cols]

X_train = train_features[feature_cols]
y_train = train_target
X_val = val_features[feature_cols]
y_val = val_target

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

# Create output directory for models
os.makedirs('models', exist_ok=True)

# Initialize ML Training Pipeline
pipeline = MLTradingPipeline()

# Prepare data for pipeline
pipeline.data = pd.concat([train_features, val_features])
pipeline.X_train = X_train
pipeline.X_val = X_val
pipeline.y_train = y_train
pipeline.y_val = y_val
pipeline.feature_cols = feature_cols

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
        print(f"\n  Training {model_name}...")
        trainer.train(model_type=model_type)

        # Evaluate on validation set
        print("\n  Evaluating on validation set...")
        val_metrics = trainer.evaluate(X_val, y_val, dataset_name='Validation')

        # Get feature importance
        importance = trainer.get_feature_importance(top_n=15)

        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'models/{model_type}_2025ytd_{timestamp}.pkl'
        trainer.save_model(model_filename)

        results[model_type] = {
            'name': model_name,
            'metrics': val_metrics,
            'feature_importance': importance,
            'model_file': model_filename
        }

        print(f"\n✓ {model_name} Training Complete")
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"  Validation Precision: {val_metrics['precision']:.3f}")
        print(f"  Validation Recall: {val_metrics['recall']:.3f}")
        print(f"  Validation F1: {val_metrics['f1']:.3f}")
        print(f"  Model saved: {model_filename}")

        if importance is not None:
            print("\n  Top 10 Features:")
            for i, (feat, imp) in enumerate(importance[:10], 1):
                print(f"    {i:2d}. {feat:30s} {imp:.4f}")

    except Exception as e:
        print(f"\n✗ Error training {model_name}: {e}")
        results[model_type] = {'error': str(e)}

# ============================================================================
# STEP 7: Model Comparison
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON (Validation Set)")
print("=" * 80)

comparison_data = []
for model_type, result in results.items():
    if 'metrics' in result:
        comparison_data.append({
            'Model': result['name'],
            'Accuracy': f"{result['metrics']['accuracy']:.3f}",
            'Precision': f"{result['metrics']['precision']:.3f}",
            'Recall': f"{result['metrics']['recall']:.3f}",
            'F1 Score': f"{result['metrics']['f1']:.3f}",
            'ROC AUC': f"{result['metrics'].get('roc_auc', 0):.3f}"
        })

if comparison_data:
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Save comparison
    comparison_file = 'models/model_comparison_2025ytd.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n✓ Comparison saved to: {comparison_file}")

# ============================================================================
# STEP 8: Save Training Summary
# ============================================================================
print("\n" + "=" * 80)
print("Saving Training Summary")
print("=" * 80)

summary = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_file': data_file,
    'data_range': f"{df.index[0]} to {df.index[-1]}",
    'total_samples': len(df),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'num_features': len(feature_cols),
    'feature_list': feature_cols,
    'train_ratio': f"{len(train_df) / (len(train_df) + len(val_df)) * 100:.1f}%",
    'val_ratio': f"{len(val_df) / (len(train_df) + len(val_df)) * 100:.1f}%",
    'target_distribution_train': {
        'up': int(train_target.sum()),
        'down': int((~train_target.astype(bool)).sum()),
        'up_pct': f"{train_target.mean() * 100:.1f}%"
    },
    'models_trained': results
}

summary_file = 'models/training_summary_2025ytd.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"✓ Training summary saved to: {summary_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

print("\n📊 Dataset: 2025 YTD S&P 500")
print(f"  Total rows: {len(df)}")
print(f"  Training: {len(X_train)} samples ({len(train_df) / (len(train_df) + len(val_df)) * 100:.0f}%)")
print(f"  Validation: {len(X_val)} samples ({len(val_df) / (len(train_df) + len(val_df)) * 100:.0f}%)")

print(f"\n🎯 Models Trained: {len([r for r in results.values() if 'metrics' in r])}")
for model_type, result in results.items():
    if 'metrics' in result:
        print(f"  ✓ {result['name']}: {result['metrics']['accuracy']:.1%} accuracy")

print("\n📁 Output Files:")
print("  Models: models/*_2025ytd_*.pkl")
print(f"  Comparison: {comparison_file}")
print(f"  Summary: {summary_file}")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Review model comparison to select best model")
print("  2. Use model in MLStrategy for backtesting")
print("  3. Consider walk-forward validation for robustness")
print("=" * 80)
