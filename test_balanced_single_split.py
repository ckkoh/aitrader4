#!/usr/bin/env python3
"""
Test Balanced Class Weights on Single Split (Split 9)
Quick verification before full validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from feature_engineering import FeatureEngineering
from ml_training_pipeline import MLModelTrainer
from backtesting_engine import BacktestEngine, BacktestConfig
from regime_adaptive_strategy import RegimeAdaptiveMLStrategy

# Top 20 features
TOP_20_FEATURES = [
    'bullish_engulfing', 'stoch_d_3', 'week_of_year', 'atr_14', 'regime',
    'roc_20', 'obv', 'parkinson_vol_10', 'volatility_200d', 'momentum_5',
    'macd_signal', 'adx_14', 'month_sin', 'hl_ratio', 'rsi_14',
    'stoch_k_14', 'bb_position_20', 'momentum_oscillator', 'pvt', 'price_acceleration'
]

def main():
    print("="*80)
    print("BALANCED MODEL TEST - Split 9 (2024 Q1 Bull Market)")
    print("="*80)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
    df = df.loc['2020-01-01':]
    print(f"  ✅ Loaded {len(df)} days")

    # 2. Generate features
    print("\n[2/6] Generating features...")
    df_features = FeatureEngineering.build_complete_feature_set(df, include_volume=True)
    df_features = df_features.dropna()
    print(f"  ✅ {len(df_features)} clean data points")

    # 3. Create Split 9
    print("\n[3/6] Creating Split 9...")
    days_per_month = 21
    train_days = 12 * days_per_month
    test_days = 3 * days_per_month
    step_days = 3 * days_per_month

    start_idx = 8 * step_days
    train_start = start_idx
    train_end = start_idx + train_days
    test_start = train_end
    test_end = test_start + test_days

    train_data = df_features.iloc[train_start:train_end]
    test_data = df_features.iloc[test_start:test_end]

    print(f"  Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
    print(f"  Test:  {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")

    # 4. Train ORIGINAL model (no balancing)
    print("\n[4/6] Training ORIGINAL model (no class balancing)...")
    df_train = train_data.copy()
    df_train['target'] = (df_train['close'].shift(-1) > df_train['close']).astype(int)
    df_train = df_train.dropna()

    X_train = df_train[TOP_20_FEATURES].values
    y_train = df_train['target'].values

    print(f"  Training samples: {len(X_train)}")
    print(f"  Class distribution: {np.bincount(y_train)} (0=SELL, 1=BUY)")
    print(f"  Class proportions: SELL={y_train[y_train==0].shape[0]/len(y_train):.1%}, BUY={y_train[y_train==1].shape[0]/len(y_train):.1%}")

    # Train original
    trainer_orig = MLModelTrainer(model_type='xgboost', task='classification')
    trainer_orig.train(X_train, y_train, hyperparameter_tuning=True)
    trainer_orig.save_model('balanced_model_results/test_split9_original.pkl')

    # Check predictions on test data
    df_test = test_data.copy()
    X_test = df_test[TOP_20_FEATURES].values
    proba_orig = trainer_orig.model.predict_proba(X_test)  # Use raw model

    print(f"\n  Original Model Predictions on Test Data:")
    print(f"    BUY probability range: {proba_orig[:, 1].min():.3f} - {proba_orig[:, 1].max():.3f}")
    print(f"    BUY probability mean: {proba_orig[:, 1].mean():.3f}")
    print(f"    SELL probability mean: {proba_orig[:, 0].mean():.3f}")
    print(f"    Predicted BUY signals: {(proba_orig[:, 1] > 0.5).sum()}/{len(proba_orig)} ({(proba_orig[:, 1] > 0.5).sum()/len(proba_orig)*100:.1f}%)")

    # 5. Train BALANCED model
    print("\n[5/6] Training BALANCED model (with class weights)...")

    # Calculate class weights
    classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=classes,
        y=y_train
    )

    # Create sample weights
    sample_weights = np.zeros(len(y_train))
    for i, cls in enumerate(classes):
        sample_weights[y_train == cls] = class_weights[i]

    print(f"  Class weights: {dict(zip(classes, class_weights))}")
    print(f"  Sample weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")

    # Train with balanced weights
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    base_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    trainer_balanced = MLModelTrainer(model_type='xgboost', task='classification')
    trainer_balanced.model = grid_search.best_estimator_
    trainer_balanced.save_model('balanced_model_results/test_split9_balanced.pkl')

    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")

    # Check predictions on test data
    proba_balanced = trainer_balanced.model.predict_proba(X_test)  # Use raw model

    print(f"\n  Balanced Model Predictions on Test Data:")
    print(f"    BUY probability range: {proba_balanced[:, 1].min():.3f} - {proba_balanced[:, 1].max():.3f}")
    print(f"    BUY probability mean: {proba_balanced[:, 1].mean():.3f}")
    print(f"    SELL probability mean: {proba_balanced[:, 0].mean():.3f}")
    print(f"    Predicted BUY signals: {(proba_balanced[:, 1] > 0.5).sum()}/{len(proba_balanced)} ({(proba_balanced[:, 1] > 0.5).sum()/len(proba_balanced)*100:.1f}%)")

    # 6. Compare predictions
    print("\n[6/6] Comparison:")
    print("="*80)

    print(f"\n📊 Prediction Distribution Shift:")
    print(f"  Original  - BUY: {proba_orig[:, 1].mean():.1%}, SELL: {proba_orig[:, 0].mean():.1%}")
    print(f"  Balanced  - BUY: {proba_balanced[:, 1].mean():.1%}, SELL: {proba_balanced[:, 0].mean():.1%}")
    print(f"  Improvement: {(proba_balanced[:, 1].mean() - proba_orig[:, 1].mean())*100:+.1f}pp more BUY predictions")

    print(f"\n📈 Signal Count at 0.50 threshold:")
    print(f"  Original:  {(proba_orig[:, 1] > 0.5).sum()} BUY, {(proba_orig[:, 0] > 0.5).sum()} SELL")
    print(f"  Balanced:  {(proba_balanced[:, 1] > 0.5).sum()} BUY, {(proba_balanced[:, 0] > 0.5).sum()} SELL")

    # Test if this would improve the zero trades issue
    print(f"\n🎯 Impact on Adaptive Strategy (BULL regime, 0.40 threshold):")
    buy_signals_orig = (proba_orig[:, 1] > 0.40).sum()
    sell_signals_orig = (proba_orig[:, 0] > 0.40).sum()
    buy_signals_balanced = (proba_balanced[:, 1] > 0.40).sum()
    sell_signals_balanced = (proba_balanced[:, 0] > 0.40).sum()

    print(f"  Original:  {buy_signals_orig} BUY signals, {sell_signals_orig} SELL signals")
    print(f"  Balanced:  {buy_signals_balanced} BUY signals, {sell_signals_balanced} SELL signals")
    print(f"  Change: {buy_signals_balanced - buy_signals_orig:+d} BUY, {sell_signals_balanced - sell_signals_orig:+d} SELL")

    if buy_signals_balanced > buy_signals_orig:
        print(f"\n  ✅ IMPROVEMENT! More BUY signals in BULL market")
    else:
        print(f"\n  ⚠️  No improvement in BUY signal count")

    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)

    # Recommendation
    if proba_balanced[:, 1].mean() > 0.40:
        print("\n✅ RECOMMENDATION: Balanced training improves BUY predictions")
        print("   Proceed with full 15-period validation using balanced models")
    else:
        print("\n⚠️  RECOMMENDATION: Balanced training shows marginal improvement")
        print("   Consider additional strategies (ensemble, regime-specific models)")


if __name__ == "__main__":
    # Create output dir
    Path('balanced_model_results').mkdir(exist_ok=True)
    main()
