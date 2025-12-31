#!/usr/bin/env python3
"""
Train ML Model with Class Balancing and Calibration

Implements three improvements:
1. Class balancing (handles imbalanced train data)
2. SMOTE oversampling (creates synthetic minority samples)
3. Probability calibration (ensures confidence scores are meaningful)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
import pickle

from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLModelTrainer

class BalancedMLTrainer:
    """
    ML Trainer with class balancing and calibration
    """

    def __init__(self, output_dir: str = 'balanced_models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load S&P 500 historical data"""
        print("\n" + "="*80)
        print("1. LOADING DATA")
        print("="*80)

        data_file = 'sp500_historical_data.csv'
        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        df = df.loc['2020-01-01':]

        print(f"✅ Loaded {len(df)} days")
        print(f"   Date range: {df.index[0]} to {df.index[-1]}")

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features"""
        print("\n" + "="*80)
        print("2. GENERATING FEATURES")
        print("="*80)

        df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)
        df_features = df_features.dropna()

        print(f"✅ Clean data points: {len(df_features)}")
        print(f"   Total features: {len(df_features.columns)}")

        return df_features

    def create_labels(self, df: pd.DataFrame, forward_days: int = 5) -> pd.Series:
        """
        Create binary classification labels
        1 = BUY (price goes up in next N days)
        0 = SELL/HOLD (price goes down or flat)
        """
        print("\n" + "="*80)
        print(f"3. CREATING LABELS (forward_days={forward_days})")
        print("="*80)

        # Calculate future return
        future_return = df['close'].pct_change(forward_days).shift(-forward_days)

        # Binary labels: 1 if positive return, 0 otherwise
        labels = (future_return > 0).astype(int)

        # Remove rows where we can't calculate future return
        labels = labels.iloc[:-forward_days]

        # Class distribution
        class_counts = labels.value_counts()
        print(f"\n📊 Class Distribution:")
        print(f"   SELL/HOLD (0): {class_counts.get(0, 0)} ({class_counts.get(0, 0) / len(labels) * 100:.1f}%)")
        print(f"   BUY (1):       {class_counts.get(1, 0)} ({class_counts.get(1, 0) / len(labels) * 100:.1f}%)")

        imbalance_ratio = class_counts.get(0, 0) / class_counts.get(1, 0) if class_counts.get(1, 0) > 0 else 0
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 1.5:
            print(f"   ⚠️  Data is imbalanced! Will apply balancing techniques.")
        else:
            print(f"   ✅ Data is reasonably balanced.")

        return labels

    def select_top_features(self, df_features: pd.DataFrame, top_n: int = 20) -> list:
        """Load or use top 20 features from feature selection"""
        print("\n" + "="*80)
        print(f"4. SELECTING TOP {top_n} FEATURES")
        print("="*80)

        # Try to load from feature selection results
        feature_file = 'feature_selection_results/top_20_features.csv'
        if Path(feature_file).exists():
            top_features_df = pd.read_csv(feature_file)
            top_features = top_features_df['feature'].tolist()
            print(f"✅ Loaded {len(top_features)} features from {feature_file}")
        else:
            # Fallback: exclude target and OHLCV columns
            exclude = ['open', 'high', 'low', 'close', 'volume',
                      'target_class', 'target_binary', 'target_regression', 'future_return']
            all_features = [col for col in df_features.columns if col not in exclude]
            top_features = all_features[:top_n]
            print(f"⚠️  Feature file not found, using first {len(top_features)} features")

        print(f"\n📋 Features:")
        for i, feat in enumerate(top_features[:10], 1):
            print(f"   {i}. {feat}")
        if len(top_features) > 10:
            print(f"   ... and {len(top_features) - 10} more")

        return top_features

    def train_model(self,
                   df_features: pd.DataFrame,
                   labels: pd.Series,
                   feature_cols: list,
                   use_smote: bool = True,
                   use_class_weight: bool = True,
                   use_calibration: bool = True) -> dict:
        """
        Train model with balancing techniques

        Args:
            df_features: Feature DataFrame
            labels: Target labels
            feature_cols: List of feature columns to use
            use_smote: Apply SMOTE oversampling
            use_class_weight: Use class weights in training
            use_calibration: Apply probability calibration

        Returns:
            dict with model, metrics, and metadata
        """
        print("\n" + "="*80)
        print("5. TRAINING MODEL")
        print("="*80)

        # Align features and labels
        df_aligned = df_features.iloc[:-5].copy()  # Remove last 5 rows (future_days)
        X = df_aligned[feature_cols].values
        y = labels.values

        print(f"\n📊 Training Data:")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Class 0: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
        print(f"   Class 1: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

        # Step 1: SMOTE Oversampling
        if use_smote:
            print(f"\n🔄 Applying SMOTE oversampling...")
            try:
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_resampled, y_resampled = smote.fit_resample(X, y)

                print(f"   Before SMOTE: {len(X)} samples")
                print(f"   After SMOTE:  {len(X_resampled)} samples")
                print(f"   Class 0: {(y_resampled == 0).sum()} ({(y_resampled == 0).sum() / len(y_resampled) * 100:.1f}%)")
                print(f"   Class 1: {(y_resampled == 1).sum()} ({(y_resampled == 1).sum() / len(y_resampled) * 100:.1f}%)")

                X_train, y_train = X_resampled, y_resampled
            except Exception as e:
                print(f"   ⚠️  SMOTE failed: {e}")
                print(f"   Continuing without SMOTE")
                X_train, y_train = X, y
        else:
            X_train, y_train = X, y

        # Step 2: Calculate Class Weights
        sample_weights = None
        if use_class_weight:
            print(f"\n⚖️  Calculating class weights...")
            class_weights = class_weight.compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            sample_weights = np.array([class_weights[int(label)] for label in y_train])

            print(f"   Class 0 weight: {class_weights[0]:.3f}")
            print(f"   Class 1 weight: {class_weights[1]:.3f}")
            print(f"   Ratio: {class_weights[1] / class_weights[0]:.2f}x")

        # Step 3: Train Base Model
        print(f"\n🤖 Training XGBoost model...")
        from xgboost import XGBClassifier
        from sklearn.model_selection import GridSearchCV

        # Create base model
        base_model = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )

        # Hyperparameter tuning with class weights
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'scale_pos_weight': [1, class_weights[1] / class_weights[0]] if use_class_weight else [1]
        }

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='f1',  # Use F1 score for imbalanced data
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train, sample_weight=sample_weights)

        best_model = grid_search.best_estimator_
        print(f"   ✅ Best parameters: {grid_search.best_params_}")
        print(f"   ✅ Best F1 score: {grid_search.best_score_:.4f}")

        # Step 4: Probability Calibration
        final_model = best_model
        if use_calibration:
            print(f"\n📏 Applying probability calibration...")
            try:
                calibrated_model = CalibratedClassifierCV(
                    best_model,
                    method='sigmoid',  # Platt scaling
                    cv=5
                )
                calibrated_model.fit(X_train, y_train)
                final_model = calibrated_model
                print(f"   ✅ Calibration complete")
            except Exception as e:
                print(f"   ⚠️  Calibration failed: {e}")
                print(f"   Using uncalibrated model")

        # Step 5: Evaluate on Training Data
        print(f"\n📊 Training Set Evaluation:")
        y_pred = final_model.predict(X_train)
        y_proba = final_model.predict_proba(X_train)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred, zero_division=0)
        recall = recall_score(y_train, y_pred, zero_division=0)
        f1 = f1_score(y_train, y_pred, zero_division=0)

        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")

        # Analyze prediction probabilities
        buy_probas = y_proba[:, 1]
        print(f"\n📈 BUY Probability Distribution:")
        print(f"   Min:    {buy_probas.min():.4f}")
        print(f"   25%:    {np.percentile(buy_probas, 25):.4f}")
        print(f"   Median: {np.median(buy_probas):.4f}")
        print(f"   75%:    {np.percentile(buy_probas, 75):.4f}")
        print(f"   Max:    {buy_probas.max():.4f}")
        print(f"   Mean:   {buy_probas.mean():.4f}")

        # Count predictions by threshold
        print(f"\n🎯 Signals at Different Thresholds:")
        for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            count = (buy_probas >= threshold).sum()
            pct = count / len(buy_probas) * 100
            print(f"   ≥{threshold:.2f}: {count:4d} ({pct:5.1f}%)")

        return {
            'model': final_model,
            'feature_cols': feature_cols,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'config': {
                'use_smote': use_smote,
                'use_class_weight': use_class_weight,
                'use_calibration': use_calibration,
                'forward_days': 5
            },
            'training_date': datetime.now().isoformat()
        }

    def save_model(self, model_data: dict, name: str = 'balanced_model'):
        """Save trained model"""
        print("\n" + "="*80)
        print("6. SAVING MODEL")
        print("="*80)

        model_path = self.output_dir / f'{name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved to: {model_path}")

        # Save metadata
        metadata_path = self.output_dir / f'{name}_metadata.json'
        import json
        metadata = {
            'feature_cols': model_data['feature_cols'],
            'metrics': model_data['metrics'],
            'config': model_data['config'],
            'training_date': model_data['training_date']
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved to: {metadata_path}")

        return model_path


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("BALANCED ML MODEL TRAINING PIPELINE")
    print("="*80)

    trainer = BalancedMLTrainer()

    # Load and prepare data
    df = trainer.load_data()
    df_features = trainer.prepare_features(df)
    labels = trainer.create_labels(df_features, forward_days=5)
    feature_cols = trainer.select_top_features(df_features, top_n=20)

    # Train model with all improvements
    model_data = trainer.train_model(
        df_features,
        labels,
        feature_cols,
        use_smote=True,
        use_class_weight=True,
        use_calibration=True
    )

    # Save model
    model_path = trainer.save_model(model_data, name='balanced_model_v1')

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel saved to: {model_path}")
    print(f"\nNext steps:")
    print(f"1. Test model predictions: python diagnose_model_predictions.py")
    print(f"2. Run walk-forward validation with new model")
    print(f"3. Compare performance vs baseline")

    return model_path


if __name__ == '__main__':
    main()
