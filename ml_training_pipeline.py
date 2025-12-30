"""
Machine Learning Training Pipeline for Trading
Includes multiple models, hyperparameter tuning, and proper time-series validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import pickle
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from feature_engineering import FeatureEngineering, DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """
    Comprehensive ML training pipeline for trading strategies
    """

    def __init__(self, model_type: str = 'xgboost', task: str = 'classification'):
        """
        Initialize ML trainer

        Args:
            model_type: 'logistic', 'random_forest', 'xgboost', 'gradient_boosting', 'ensemble'
            task: 'classification' or 'regression'
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.training_history = {}
        self.selected_features = None

    def prepare_data(self, df: pd.DataFrame,
                     feature_cols: List[str],
                     target_col: str = 'target_binary',
                     test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training with proper time-series split

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            test_size: Proportion of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, train_dates, test_dates)
        """
        # Remove NaNs
        df = df.dropna(subset=feature_cols + [target_col])

        # Time-series split (no shuffling!)
        split_idx = int(len(df) * (1 - test_size))

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        train_dates = train_df.index
        test_dates = test_df.index

        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Training period: {train_dates[0]} to {train_dates[-1]}")
        logger.info(f"Test period: {test_dates[0]} to {test_dates[-1]}")

        return X_train, X_test, y_train, y_test, train_dates, test_dates

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              hyperparameter_tuning: bool = False) -> Dict:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")

        # Normalize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Get model
        if hyperparameter_tuning:
            self.model = self._train_with_tuning(X_train_scaled, y_train)
        else:
            self.model = self._get_model()

            if self.model_type == 'xgboost' and X_val is not None:
                # XGBoost with early stopping (newer API)
                try:
                    # Try new API (XGBoost >= 2.0)
                    self.model.fit(
                        X_train_scaled, y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        verbose=False
                    )
                except TypeError:
                    # Fallback for older XGBoost versions
                    self.model.fit(X_train_scaled, y_train)
            else:
                self.model.fit(X_train_scaled, y_train)

        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_[0])

        # Validation metrics
        metrics = {}
        if X_val is not None:
            val_pred = self.predict(X_val)
            metrics = self._calculate_metrics(y_val, val_pred)
            logger.info(f"Validation Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Validation F1: {metrics['f1_score']:.4f}")

        return metrics

    def _get_model(self):
        """Get model instance based on model_type"""
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )

        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )

        elif self.model_type == 'xgboost':
            if self.task == 'classification':
                return xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                )

        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        elif self.model_type == 'ensemble':
            # Voting ensemble
            models = [
                ('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('r', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42))
            ]
            return VotingClassifier(estimators=models, voting='soft', n_jobs=-1)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _train_with_tuning(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train with hyperparameter tuning"""
        logger.info("Performing hyperparameter tuning...")

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        if self.model_type == 'xgboost':
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            base_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [10, 20, 50],
                'min_samples_lea': [5, 10, 20]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

        else:
            # Default to basic model
            return self._get_model()

        # Use RandomizedSearchCV for faster tuning
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=20,
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")

        return random_search.best_estimator_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For models without predict_proba, return binary predictions
            predictions = self.model.predict(X_scaled)
            return np.column_stack([1 - predictions, predictions])

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: True test labels

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        metrics = self._calculate_metrics(y_test, predictions)

        # Print detailed results
        logger.info("\n" + "=" * 50)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")

        if 'roc_auc' in metrics:
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")

        logger.info("\nConfusion Matrix:")
        logger.info(metrics['confusion_matrix'])
        logger.info("=" * 50 + "\n")

        return metrics

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2:
            try:
                y_proba = self.predict_proba(self.scaler.transform(
                    np.zeros((len(y_pred), self.scaler.n_features_in_))
                ))[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except BaseException:
                pass

        return metrics

    def cross_validate(self, df: pd.DataFrame,
                       feature_cols: List[str],
                       target_col: str = 'target_binary',
                       n_splits: int = 5) -> Dict:
        """
        Perform time-series cross-validation

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            n_splits: Number of CV splits

        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {n_splits}-fold time-series cross-validation...")

        df = df.dropna(subset=feature_cols + [target_col])
        X = df[feature_cols].values
        y = df[target_col].values

        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"Fold {fold}/{n_splits}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train fold model
            fold_model = self._get_model()
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            fold_model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = fold_model.predict(X_val_scaled)

            cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
            cv_scores['precision'].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
            cv_scores['recall'].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
            cv_scores['f1'].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))

        # Calculate mean and std
        results = {}
        for metric, scores in cv_scores.items():
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)

        logger.info("\nCross-Validation Results:")
        logger.info(f"Accuracy: {results['accuracy_mean']:.4f} (+/- {results['accuracy_std']:.4f})")
        logger.info(f"F1 Score: {results['f1_mean']:.4f} (+/- {results['f1_std']:.4f})")

        return results

    def get_feature_importance(self, feature_names: List[str],
                               top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance ranking

        Args:
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not available for this model")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'task': self.task,
            'feature_importance': self.feature_importance,
            'selected_features': self.selected_features,
            'training_date': datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> 'MLModelTrainer':
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        trainer = MLModelTrainer(
            model_type=model_data['model_type'],
            task=model_data['task']
        )

        trainer.model = model_data['model']
        trainer.scaler = model_data['scaler']
        trainer.feature_importance = model_data['feature_importance']
        trainer.selected_features = model_data['selected_features']

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Training date: {model_data['training_date']}")

        return trainer


class MLTradingPipeline:
    """
    Complete ML pipeline for trading strategy development
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.df_features = None
        self.feature_cols = None
        self.trainer = None
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)

    def load_and_prepare_data(self, df: pd.DataFrame,
                              include_volume: bool = False) -> pd.DataFrame:
        """
        Load and prepare data with features

        Args:
            df: DataFrame with OHLCV data
            include_volume: Whether to include volume features

        Returns:
            DataFrame with features
        """
        logger.info("Preparing data and engineering features...")

        # Build feature set
        self.df_features = FeatureEngineering.build_complete_feature_set(
            df, include_volume=include_volume
        )

        # Clean data
        self.df_features = DataPreprocessor.clean_data(self.df_features)

        # Get feature columns (exclude OHLCV and targets)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                        'future_return', 'target_class', 'target_binary',
                        'target_regression']
        self.feature_cols = [col for col in self.df_features.columns
                             if col not in exclude_cols]

        logger.info(f"Total features: {len(self.feature_cols)}")

        return self.df_features

    def train_model(self, model_type: str = 'xgboost',
                    target_col: str = 'target_binary',
                    test_size: float = 0.2,
                    hyperparameter_tuning: bool = False,
                    cross_validation: bool = True) -> Dict:
        """
        Train ML model

        Args:
            model_type: Type of model to train
            target_col: Target column name
            test_size: Test set proportion
            hyperparameter_tuning: Whether to tune hyperparameters
            cross_validation: Whether to perform CV

        Returns:
            Dictionary with training results
        """
        if self.df_features is None:
            raise ValueError("Data not prepared. Call load_and_prepare_data first.")

        # Initialize trainer
        self.trainer = MLModelTrainer(model_type=model_type, task='classification')

        # Prepare data
        X_train, X_test, y_train, y_test, train_dates, test_dates = \
            self.trainer.prepare_data(
                self.df_features,
                self.feature_cols,
                target_col,
                test_size
            )

        # Cross-validation
        if cross_validation:
            cv_results = self.trainer.cross_validate(
                self.df_features,
                self.feature_cols,
                target_col,
                n_splits=5
            )

        # Split train into train/validation
        val_split = int(len(X_train) * 0.8)
        X_train_split = X_train[:val_split]
        y_train_split = y_train[:val_split]
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]

        # Train
        train_metrics = self.trainer.train(
            X_train_split, y_train_split,
            X_val, y_val,
            hyperparameter_tuning=hyperparameter_tuning
        )

        # Evaluate on test set
        test_metrics = self.trainer.evaluate(X_test, y_test)

        # Feature importance
        feature_importance = self.trainer.get_feature_importance(
            self.feature_cols, top_n=30
        )

        logger.info("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")

        # Save model
        model_filename = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.trainer.save_model(str(self.models_dir / model_filename))

        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'model_filename': model_filename
        }

        if cross_validation:
            results['cv_results'] = cv_results

        return results

    def compare_models(self, model_types: List[str] = None,
                       target_col: str = 'target_binary') -> pd.DataFrame:
        """
        Compare multiple models

        Args:
            model_types: List of model types to compare
            target_col: Target column name

        Returns:
            DataFrame with comparison results
        """
        if model_types is None:
            model_types = ['logistic', 'random_forest', 'xgboost', 'gradient_boosting']

        logger.info(f"Comparing {len(model_types)} models...")

        results = []

        for model_type in model_types:
            logger.info(f"\nTraining {model_type}...")

            try:
                result = self.train_model(
                    model_type=model_type,
                    target_col=target_col,
                    hyperparameter_tuning=False,
                    cross_validation=False
                )

                test_metrics = result['test_metrics']
                results.append({
                    'model': model_type,
                    'accuracy': test_metrics['accuracy'],
                    'precision': test_metrics['precision'],
                    'recall': test_metrics['recall'],
                    'f1_score': test_metrics['f1_score']
                })

            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")

        comparison_df = pd.DataFrame(results).sort_values('f1_score', ascending=False)

        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("=" * 60)
        logger.info(comparison_df.to_string(index=False))
        logger.info("=" * 60 + "\n")

        return comparison_df


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2022-01-01', periods=5000, freq='1H')
    np.random.seed(42)

    # Simulate realistic price data
    returns = np.random.randn(5000) * 0.001
    price = 1.1 + np.cumsum(returns)

    df = pd.DataFrame({
        'open': price + np.random.randn(5000) * 0.0001,
        'high': price + np.abs(np.random.randn(5000)) * 0.0002,
        'low': price - np.abs(np.random.randn(5000)) * 0.0002,
        'close': price,
        'volume': np.random.randint(1000, 10000, 5000)
    }, index=dates)

    # Initialize pipeline
    pipeline = MLTradingPipeline()

    # Prepare data
    df_features = pipeline.load_and_prepare_data(df, include_volume=True)

    # Train single model
    results = pipeline.train_model(
        model_type='xgboost',
        hyperparameter_tuning=False,
        cross_validation=True
    )

    # Compare models
    comparison = pipeline.compare_models()
