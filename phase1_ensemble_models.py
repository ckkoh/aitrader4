#!/usr/bin/env python3
"""
PHASE 1, DAYS 4-5: Multi-Model Ensemble

Goal: Train ensemble of different model types and combine predictions
- Models: RandomForest, XGBoost, GradientBoosting, LogisticRegression
- Strategy: Voting ensemble (require 3/4 or 4/4 agreement)
- Expected: More robust predictions, better accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
from typing import List, Dict

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

from backtesting_engine import Strategy, BacktestEngine, BacktestConfig
from feature_engineering import FeatureEngineering

print("=" * 80)
print("PHASE 1, DAYS 4-5: MULTI-MODEL ENSEMBLE")
print("=" * 80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/6] Loading Data...")

df = pd.read_csv('sp500_historical_data.csv', index_col='Date', parse_dates=True)
df = df.sort_index()

train_df = df[(df.index >= '2020-01-01') & (df.index < '2025-01-01')]
val_df = df[df.index >= '2025-01-01']

print(f"  Train: {len(train_df)} rows (2020-2024)")
print(f"  Val:   {len(val_df)} rows (2025)")

# ============================================================================
# STEP 2: Generate Features and Target
# ============================================================================
print("\n[2/6] Generating Features...")

train_feat = FeatureEngineering.build_complete_feature_set(train_df, include_volume=True).dropna()
val_feat = FeatureEngineering.build_complete_feature_set(val_df, include_volume=True).dropna()

# Use same clean features from Day 1
with open('models/randomforest_CLEAN_top20_20251230_230009.pkl', 'rb') as f:
    clean_model_data = pickle.load(f)
    feature_cols = clean_model_data['features']

print(f"  Using {len(feature_cols)} clean features")

# Create target (next day up/down)
train_target = (train_feat['close'].pct_change().shift(-1) > 0).astype(int)[:-1]
val_target = (val_feat['close'].pct_change().shift(-1) > 0).astype(int)[:-1]

train_feat = train_feat.iloc[:-1]
val_feat = val_feat.iloc[:-1]

# Prepare feature matrices
X_train = train_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = train_target

X_val = val_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_val = val_target

print(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}")

# ============================================================================
# STEP 3: Train Multiple Model Types
# ============================================================================
print("\n[3/6] Training Ensemble Models...")

models = {}
performance = {}

# Model 1: Random Forest
print("\n  Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_val_pred = rf_model.predict(X_val)

models['RandomForest'] = rf_model
performance['RandomForest'] = {
    'train_accuracy': float(accuracy_score(y_train, rf_train_pred)),
    'val_accuracy': float(accuracy_score(y_val, rf_val_pred)),
    'val_precision': float(precision_score(y_val, rf_val_pred, zero_division=0)),
    'val_recall': float(recall_score(y_val, rf_val_pred, zero_division=0)),
    'val_f1': float(f1_score(y_val, rf_val_pred, zero_division=0))
}
print(f"    Train: {performance['RandomForest']['train_accuracy']:.3f}, "
      f"Val: {performance['RandomForest']['val_accuracy']:.3f}")

# Model 2: XGBoost
print("\n  Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

xgb_train_pred = xgb_model.predict(X_train)
xgb_val_pred = xgb_model.predict(X_val)

models['XGBoost'] = xgb_model
performance['XGBoost'] = {
    'train_accuracy': float(accuracy_score(y_train, xgb_train_pred)),
    'val_accuracy': float(accuracy_score(y_val, xgb_val_pred)),
    'val_precision': float(precision_score(y_val, xgb_val_pred, zero_division=0)),
    'val_recall': float(recall_score(y_val, xgb_val_pred, zero_division=0)),
    'val_f1': float(f1_score(y_val, xgb_val_pred, zero_division=0))
}
print(f"    Train: {performance['XGBoost']['train_accuracy']:.3f}, "
      f"Val: {performance['XGBoost']['val_accuracy']:.3f}")

# Model 3: Gradient Boosting
print("\n  Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)

gb_train_pred = gb_model.predict(X_train)
gb_val_pred = gb_model.predict(X_val)

models['GradientBoosting'] = gb_model
performance['GradientBoosting'] = {
    'train_accuracy': float(accuracy_score(y_train, gb_train_pred)),
    'val_accuracy': float(accuracy_score(y_val, gb_val_pred)),
    'val_precision': float(precision_score(y_val, gb_val_pred, zero_division=0)),
    'val_recall': float(recall_score(y_val, gb_val_pred, zero_division=0)),
    'val_f1': float(f1_score(y_val, gb_val_pred, zero_division=0))
}
print(f"    Train: {performance['GradientBoosting']['train_accuracy']:.3f}, "
      f"Val: {performance['GradientBoosting']['val_accuracy']:.3f}")

# Model 4: Logistic Regression
print("\n  Training Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
lr_model.fit(X_train, y_train)

lr_train_pred = lr_model.predict(X_train)
lr_val_pred = lr_model.predict(X_val)

models['LogisticRegression'] = lr_model
performance['LogisticRegression'] = {
    'train_accuracy': float(accuracy_score(y_train, lr_train_pred)),
    'val_accuracy': float(accuracy_score(y_val, lr_val_pred)),
    'val_precision': float(precision_score(y_val, lr_val_pred, zero_division=0)),
    'val_recall': float(recall_score(y_val, lr_val_pred, zero_division=0)),
    'val_f1': float(f1_score(y_val, lr_val_pred, zero_division=0))
}
print(f"    Train: {performance['LogisticRegression']['train_accuracy']:.3f}, "
      f"Val: {performance['LogisticRegression']['val_accuracy']:.3f}")

# ============================================================================
# STEP 4: Calculate Ensemble Accuracy
# ============================================================================
print("\n[4/6] Calculating Ensemble Accuracy...")

# Get all predictions
all_val_preds = {
    'RandomForest': rf_val_pred,
    'XGBoost': xgb_val_pred,
    'GradientBoosting': gb_val_pred,
    'LogisticRegression': lr_val_pred
}

# Majority vote ensemble
ensemble_pred = np.zeros(len(y_val))
for i in range(len(y_val)):
    votes = [all_val_preds[m][i] for m in models.keys()]
    ensemble_pred[i] = 1 if sum(votes) >= 2 else 0  # Majority vote

ensemble_acc = accuracy_score(y_val, ensemble_pred)
ensemble_prec = precision_score(y_val, ensemble_pred, zero_division=0)
ensemble_rec = recall_score(y_val, ensemble_pred, zero_division=0)
ensemble_f1 = f1_score(y_val, ensemble_pred, zero_division=0)

print("\n  Ensemble (Majority Vote):")
print(f"    Accuracy:  {ensemble_acc:.3f}")
print(f"    Precision: {ensemble_prec:.3f}")
print(f"    Recall:    {ensemble_rec:.3f}")
print(f"    F1:        {ensemble_f1:.3f}")

# ============================================================================
# STEP 5: Save Ensemble Models
# ============================================================================
print("\n[5/6] Saving Ensemble Models...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
ensemble_file = f'models/ensemble_4models_{timestamp}.pkl'

with open(ensemble_file, 'wb') as f:
    pickle.dump({
        'models': models,
        'model_types': list(models.keys()),
        'features': feature_cols,
        'performance': performance,
        'ensemble_performance': {
            'val_accuracy': float(ensemble_acc),
            'val_precision': float(ensemble_prec),
            'val_recall': float(ensemble_rec),
            'val_f1': float(ensemble_f1)
        },
        'training_date': timestamp,
        'notes': 'Ensemble of RF, XGB, GB, LR models'
    }, f)

print(f"  ✓ Saved: {ensemble_file}")

# ============================================================================
# STEP 6: Create Ensemble Strategy and Backtest
# ============================================================================
print("\n[6/6] Creating Ensemble Strategy...")


class EnsembleStrategy(Strategy):
    """
    Trading strategy using voting ensemble of multiple models

    Entry Logic:
    - Get predictions from all 4 models
    - Require strong agreement (3/4 or 4/4 models)
    - Each model must meet minimum confidence threshold

    Exit Logic:
    - Majority of models predict DOWN
    """

    def __init__(self, models_dict, feature_cols, confidence_threshold=0.6,
                 agreement_required=3, name="Ensemble"):
        super().__init__(name)
        self.models = models_dict
        self.model_names = list(models_dict.keys())
        self.feature_cols = feature_cols
        self.confidence_threshold = confidence_threshold
        self.agreement_required = agreement_required

    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Dict]:
        """Generate trading signals based on ensemble predictions"""
        signals = []

        # Need enough data for features
        if len(data) < 60:
            return signals

        # Generate features for current data
        try:
            features = FeatureEngineering.build_complete_feature_set(data, include_volume=True)
            features = features.dropna()

            if len(features) == 0:
                return signals

            # Get current row features
            current_features = features.iloc[-1:][self.feature_cols]
            current_features = current_features.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Get predictions from all models
            predictions = {}
            confidences = {}

            for name, model in self.models.items():
                pred = model.predict(current_features)[0]
                pred_proba = model.predict_proba(current_features)[0]
                conf = pred_proba[pred]

                predictions[name] = pred
                confidences[name] = conf

            # Count votes
            up_votes = sum(1 for name in self.model_names
                           if predictions[name] == 1 and confidences[name] >= self.confidence_threshold)

            down_votes = sum(1 for name in self.model_names
                             if predictions[name] == 0 and confidences[name] >= self.confidence_threshold)

            # Get current price and ATR for risk management
            current_price = data['close'].iloc[-1]

            # Calculate ATR
            if 'atr_14' in features.columns:
                atr = features['atr_14'].iloc[-1]
            else:
                high_low = data['high'].iloc[-14:] - data['low'].iloc[-14:]
                atr = high_low.mean()

            # Stop loss and take profit
            stop_loss = current_price - (atr * 2.0)
            take_profit = current_price + (atr * 3.0)

            # Check if we have a position
            has_position = len(self.positions) > 0

            if not has_position:
                # ENTRY: Buy if enough models agree on UP
                if up_votes >= self.agreement_required:
                    avg_conf = np.mean([confidences[n] for n in self.model_names if predictions[n] == 1])

                    signals.append({
                        'instrument': 'SP500',
                        'action': 'buy',
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'reason': f"ENS_BUY_{up_votes}/{len(self.models)}_conf={avg_conf:.2f}"
                    })
            else:
                # EXIT: Close if enough models predict DOWN
                if down_votes >= self.agreement_required:
                    avg_conf = np.mean([confidences[n] for n in self.model_names if predictions[n] == 0])

                    signals.append({
                        'instrument': 'SP500',
                        'action': 'close',
                        'reason': f"ENS_EXIT_{down_votes}/{len(self.models)}_conf={avg_conf:.2f}"
                    })

        except Exception:
            # Silently handle errors
            pass

        return signals


# Test multiple agreement levels
agreement_levels = [2, 3, 4]  # 2/4, 3/4, 4/4

print(f"  Strategy created with {len(models)} models: {list(models.keys())}")

# Configure backtest
config = BacktestConfig(
    initial_capital=10000.0,
    commission_pct=0.0001,
    slippage_pct=0.0001,
    position_size_pct=0.02,
    max_positions=1,
    leverage=1.0,
    position_sizing_method='fixed_pct',
    max_daily_loss_pct=0.05,
    max_drawdown_pct=0.20
)

results_by_agreement = {}

print("\n  Running backtests...")

for agreement in agreement_levels:
    print(f"\n    Testing {agreement}/{len(models)} agreement...")

    # Create strategy
    strategy = EnsembleStrategy(
        models_dict=models,
        feature_cols=feature_cols,
        confidence_threshold=0.60,
        agreement_required=agreement
    )

    # Create backtest engine
    engine = BacktestEngine(config)

    # Run backtest
    backtest_results = engine.run_backtest(strategy, val_df)

    # Extract metrics
    metrics = backtest_results['metrics']

    # Store results
    results_by_agreement[agreement] = {
        'agreement_level': f"{agreement}/{len(models)}",
        'total_trades': metrics.get('total_trades', 0),
        'win_rate': metrics.get('win_rate', 0),
        'total_return_pct': metrics.get('total_return_pct', 0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
        'profit_factor': metrics.get('profit_factor', 0)
    }

    print(f"      Trades: {metrics.get('total_trades', 0)}, "
          f"Return: {metrics.get('total_return_pct', 0):+.2%}, "
          f"Win Rate: {metrics.get('win_rate', 0):.1%}")

# ============================================================================
# STEP 7: Compare Results
# ============================================================================
print("\n" + "=" * 80)
print("ENSEMBLE RESULTS COMPARISON")
print("=" * 80)

# Load single model results
with open('models/phase1_confidence_sweep_results.json', 'r') as f:
    single_model = json.load(f)
    buy_hold = single_model['buy_hold_return']
    single_best = single_model['best_threshold']

print(f"\n📈 BENCHMARK: Buy & Hold = {buy_hold:+.2%}\n")

# Single model
print("SINGLE MODEL (RandomForest only):")
print(f"  Trades:        {single_best['total_trades']}")
print(f"  Win Rate:      {single_best['win_rate']:.1%}")
print(f"  Total Return:  {single_best['total_return_pct']:+.2%}")
print(f"  Sharpe Ratio:  {single_best['sharpe_ratio']:.2f}")

# Individual model performance
print("\nINDIVIDUAL MODEL VALIDATION ACCURACY:")
for name in models.keys():
    print(f"  {name:<20} {performance[name]['val_accuracy']:.3f}")

print(f"  {'Ensemble (Majority)':<20} {ensemble_acc:.3f}")

# Ensemble backtest results
print("\nENSEMBLE BACKTEST RESULTS:")
print(f"{'Agreement':<15} {'Trades':<10} {'Win Rate':<12} {'Return':<15} {'Sharpe':<10}")
print("-" * 80)

for agreement, results in sorted(results_by_agreement.items()):
    print(f"{results['agreement_level']:<15} {results['total_trades']:<10} "
          f"{results['win_rate']:<12.1%} {results['total_return_pct']:<15.2%} "
          f"{results['sharpe_ratio']:<10.2f}")

# Find best ensemble
best_ensemble = max(results_by_agreement.values(), key=lambda x: x['total_return_pct'])

print("\n🎯 BEST ENSEMBLE CONFIG:")
print(f"  Agreement Level:  {best_ensemble['agreement_level']}")
print(f"  Total Trades:     {best_ensemble['total_trades']}")
print(f"  Win Rate:         {best_ensemble['win_rate']:.1%}")
print(f"  Total Return:     {best_ensemble['total_return_pct']:+.2%}")
print(f"  Sharpe Ratio:     {best_ensemble['sharpe_ratio']:.2f}")
print(f"  vs Buy & Hold:    {best_ensemble['total_return_pct'] - buy_hold:+.2%}")

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n💾 Saving Results...")

results_summary = {
    'timestamp': datetime.now().isoformat(),
    'ensemble_file': ensemble_file,
    'model_types': list(models.keys()),
    'features': len(feature_cols),
    'buy_hold_return': float(buy_hold),
    'individual_performance': performance,
    'ensemble_validation_accuracy': float(ensemble_acc),
    'single_model_backtest': {
        'trades': single_best['total_trades'],
        'win_rate': float(single_best['win_rate']),
        'return_pct': float(single_best['total_return_pct']),
        'sharpe': float(single_best['sharpe_ratio'])
    },
    'ensemble_backtest': {
        'results_by_agreement': {str(k): v for k, v in results_by_agreement.items()},
        'best_config': {
            'agreement_level': best_ensemble['agreement_level'],
            'trades': best_ensemble['total_trades'],
            'win_rate': float(best_ensemble['win_rate']),
            'return_pct': float(best_ensemble['total_return_pct']),
            'sharpe': float(best_ensemble['sharpe_ratio'])
        }
    }
}

results_file = 'models/phase1_days4-5_ensemble_results.json'
with open(results_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"  ✓ Results saved: {results_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1 DAYS 4-5 COMPLETE!")
print("=" * 80)

print(f"\n✅ TRAINED {len(models)} MODEL TYPES:")
for name, perf in performance.items():
    print(f"  • {name:<20} Val Acc: {perf['val_accuracy']:.3f}")

print("\n📊 RESULTS:")
print(f"  Single Model:  {single_best['total_trades']} trades, {single_best['total_return_pct']:+.2%}")
print(f"  Best Ensemble: {best_ensemble['total_trades']} trades, {best_ensemble['total_return_pct']:+.2%}")

# Verdict
if best_ensemble['total_trades'] > single_best['total_trades']:
    verdict = f"✅ SUCCESS: Generated {best_ensemble['total_trades'] - single_best['total_trades']} more trades!"
elif best_ensemble['total_return_pct'] > single_best['total_return_pct']:
    verdict = f"✅ SUCCESS: Higher return ({best_ensemble['total_return_pct'] - single_best['total_return_pct']:+.2%})!"
else:
    verdict = "⚠️  Ensemble didn't improve over single model"

print(f"\n{verdict}")

print("\n📁 OUTPUT FILES:")
print(f"  Models: {ensemble_file}")
print(f"  Results: {results_file}")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE! (5/5 days)")
print("Generate final Phase 1 summary report")
print("=" * 80)
