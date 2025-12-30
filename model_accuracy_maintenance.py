"""
Model Accuracy Maintenance System
Ensures ML models stay accurate over time through monitoring, drift detection, and automated retraining
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelHealthReport:
    """Comprehensive model health assessment"""
    timestamp: datetime
    model_name: str

    # Performance metrics
    current_accuracy: float
    baseline_accuracy: float
    accuracy_degradation: float

    # Drift metrics
    feature_drift_score: float
    prediction_drift_score: float
    concept_drift_detected: bool

    # Data quality
    missing_data_pct: float
    data_quality_score: float

    # Recommendations
    needs_retraining: bool
    confidence_level: str  # 'high', 'medium', 'low'
    recommended_action: str

    # Details
    warnings: List[str]
    metrics_details: Dict


class ModelAccuracyMonitor:
    """
    Continuous monitoring of model accuracy with automatic drift detection
    """

    def __init__(self, model_name: str, baseline_metrics: Dict):
        """
        Initialize accuracy monitor

        Args:
            model_name: Name of the model being monitored
            baseline_metrics: Metrics from validation/backtesting
                - accuracy: float
                - f1_score: float
                - precision: float
                - recall: float
                - feature_means: Dict[str, float]
                - feature_stds: Dict[str, float]
        """
        self.model_name = model_name
        self.baseline_metrics = baseline_metrics

        # Store historical predictions for analysis
        self.prediction_history = []
        self.performance_history = []

        # Thresholds
        self.accuracy_degradation_threshold = 0.10  # 10% drop
        self.drift_threshold = 0.15  # 15% feature drift
        self.min_samples_for_eval = 30  # Need 30 predictions minimum

    def check_accuracy(self, true_labels: np.ndarray,
                       predictions: np.ndarray,
                       prediction_probas: Optional[np.ndarray] = None) -> ModelHealthReport:
        """
        Check current model accuracy against baseline

        Args:
            true_labels: Actual outcomes
            predictions: Model predictions
            prediction_probas: Prediction probabilities (optional)

        Returns:
            ModelHealthReport with comprehensive assessment
        """
        warnings = []

        # Calculate current metrics
        current_accuracy = accuracy_score(true_labels, predictions)
        current_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

        # Compare to baseline
        baseline_accuracy = self.baseline_metrics.get('accuracy', 0.55)
        accuracy_degradation = (baseline_accuracy - current_accuracy) / baseline_accuracy

        # Store performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'accuracy': current_accuracy,
            'f1_score': current_f1,
            'sample_size': len(predictions)
        })

        # Check for significant degradation
        if accuracy_degradation > self.accuracy_degradation_threshold:
            warnings.append(
                f"Accuracy degraded by {accuracy_degradation:.1%} "
                f"(from {baseline_accuracy:.1%} to {current_accuracy:.1%})"
            )

        # Statistical significance test
        is_significant = self._test_significance(
            current_accuracy,
            baseline_accuracy,
            len(predictions)
        )

        if is_significant:
            warnings.append("Performance change is statistically significant")

        # Check prediction distribution
        prediction_drift = self._check_prediction_drift(predictions)

        if prediction_drift > 0.2:
            warnings.append(f"Prediction distribution drift: {prediction_drift:.1%}")

        # Determine recommendations
        needs_retraining = False
        confidence_level = 'high'
        recommended_action = 'continue_monitoring'

        if accuracy_degradation > 0.15 or prediction_drift > 0.3:
            needs_retraining = True
            confidence_level = 'low'
            recommended_action = 'retrain_immediately'
        elif accuracy_degradation > 0.10 or prediction_drift > 0.2:
            needs_retraining = True
            confidence_level = 'medium'
            recommended_action = 'schedule_retraining'
        elif accuracy_degradation > 0.05:
            confidence_level = 'medium'
            recommended_action = 'monitor_closely'

        return ModelHealthReport(
            timestamp=datetime.now(),
            model_name=self.model_name,
            current_accuracy=current_accuracy,
            baseline_accuracy=baseline_accuracy,
            accuracy_degradation=accuracy_degradation,
            feature_drift_score=0.0,  # Calculated separately
            prediction_drift_score=prediction_drift,
            concept_drift_detected=is_significant and accuracy_degradation > 0.10,
            missing_data_pct=0.0,  # Calculated separately
            data_quality_score=1.0,  # Calculated separately
            needs_retraining=needs_retraining,
            confidence_level=confidence_level,
            recommended_action=recommended_action,
            warnings=warnings,
            metrics_details={
                'current_accuracy': current_accuracy,
                'current_f1': current_f1,
                'baseline_accuracy': baseline_accuracy,
                'samples_evaluated': len(predictions),
                'statistically_significant': is_significant
            }
        )

    def _test_significance(self, current_acc: float, baseline_acc: float,
                           n_samples: int) -> bool:
        """
        Test if accuracy change is statistically significant
        Uses binomial test
        """
        if n_samples < self.min_samples_for_eval:
            return False

        # Binomial test
        n_correct = int(current_acc * n_samples)
        p_value = stats.binom_test(n_correct, n_samples, baseline_acc, alternative='two-sided')

        return p_value < 0.05  # 95% confidence

    def _check_prediction_drift(self, predictions: np.ndarray) -> float:
        """
        Check if prediction distribution has drifted

        Returns:
            Drift score (0-1, higher = more drift)
        """
        # Store recent predictions
        self.prediction_history.extend(predictions.tolist())

        # Keep only recent history (last 1000 predictions)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]

        if len(self.prediction_history) < 200:
            return 0.0  # Not enough history

        # Compare recent vs historical distribution
        recent = np.array(self.prediction_history[-100:])
        historical = np.array(self.prediction_history[-500:-100])

        recent_positive_rate = np.mean(recent)
        historical_positive_rate = np.mean(historical)

        drift = abs(recent_positive_rate - historical_positive_rate)

        return drift


class FeatureDriftDetector:
    """
    Detect when feature distributions change (covariate shift)
    """

    def __init__(self, baseline_features: pd.DataFrame):
        """
        Initialize with baseline feature statistics

        Args:
            baseline_features: DataFrame with features from training data
        """
        self.feature_stats = {}

        for col in baseline_features.columns:
            self.feature_stats[col] = {
                'mean': baseline_features[col].mean(),
                'std': baseline_features[col].std(),
                'min': baseline_features[col].min(),
                'max': baseline_features[col].max(),
                'q25': baseline_features[col].quantile(0.25),
                'q50': baseline_features[col].quantile(0.50),
                'q75': baseline_features[col].quantile(0.75)
            }

    def detect_drift(self, current_features: pd.DataFrame,
                     method: str = 'psi') -> Dict[str, float]:
        """
        Detect feature drift using various methods

        Args:
            current_features: Recent feature data
            method: 'psi' (Population Stability Index) or 'ks' (Kolmogorov-Smirnov)

        Returns:
            Dictionary with drift scores per feature
        """
        drift_scores = {}

        for col in current_features.columns:
            if col not in self.feature_stats:
                continue

            if method == 'psi':
                drift_scores[col] = self._calculate_psi(
                    current_features[col].values,
                    self.feature_stats[col]
                )
            elif method == 'ks':
                drift_scores[col] = self._calculate_ks(
                    current_features[col].values,
                    self.feature_stats[col]
                )

        return drift_scores

    def _calculate_psi(self, current: np.ndarray, baseline_stats: Dict) -> float:
        """
        Calculate Population Stability Index (PSI)

        PSI measures the change in distribution:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change
        """
        # Create bins based on baseline quartiles
        bins = [
            -np.inf,
            baseline_stats['q25'],
            baseline_stats['q50'],
            baseline_stats['q75'],
            np.inf
        ]

        # Baseline distribution (uniform across quartiles)
        baseline_pct = np.array([0.25, 0.25, 0.25, 0.25])

        # Current distribution
        current_counts = np.histogram(current, bins=bins)[0]
        current_pct = current_counts / len(current)

        # Avoid division by zero
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)

        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return psi

    def _calculate_ks(self, current: np.ndarray, baseline_stats: Dict) -> float:
        """
        Calculate Kolmogorov-Smirnov statistic
        Measures maximum distance between cumulative distributions
        """
        # Generate baseline sample from statistics (approximate)
        baseline_sample = np.random.normal(
            baseline_stats['mean'],
            baseline_stats['std'],
            len(current)
        )

        # KS test
        ks_stat, p_value = stats.ks_2samp(current, baseline_sample)

        return ks_stat

    def get_drift_summary(self, drift_scores: Dict[str, float],
                          threshold: float = 0.15) -> Dict:
        """
        Summarize drift detection results

        Args:
            drift_scores: Drift scores per feature
            threshold: Threshold for significant drift

        Returns:
            Summary with drifted features and overall score
        """
        drifted_features = [
            (feature, score)
            for feature, score in drift_scores.items()
            if score > threshold
        ]

        # Sort by drift score
        drifted_features.sort(key=lambda x: x[1], reverse=True)

        # Overall drift score (average of top 10 features)
        if drift_scores:
            top_scores = sorted(drift_scores.values(), reverse=True)[:10]
            overall_drift = np.mean(top_scores)
        else:
            overall_drift = 0.0

        return {
            'overall_drift_score': overall_drift,
            'drifted_features': drifted_features,
            'total_features': len(drift_scores),
            'features_drifted': len(drifted_features),
            'drift_threshold': threshold,
            'needs_attention': overall_drift > threshold
        }


class DataQualityMonitor:
    """
    Monitor data quality issues that affect model accuracy
    """

    @staticmethod
    def check_data_quality(data: pd.DataFrame,
                           expected_columns: List[str]) -> Dict:
        """
        Comprehensive data quality check

        Args:
            data: Current data
            expected_columns: List of expected feature columns

        Returns:
            Dictionary with quality metrics
        """
        issues = []

        # 1. Missing columns
        missing_cols = set(expected_columns) - set(data.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # 2. Missing values
        missing_pct = data.isnull().sum() / len(data)
        high_missing = missing_pct[missing_pct > 0.05]  # > 5% missing

        if len(high_missing) > 0:
            issues.append(f"High missing values in: {high_missing.to_dict()}")

        # 3. Constant features (no variance)
        constant_features = []
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].std() == 0:
                constant_features.append(col)

        if constant_features:
            issues.append(f"Constant features: {constant_features}")

        # 4. Outliers (beyond 5 standard deviations)
        outlier_counts = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            mean = data[col].mean()
            std = data[col].std()
            if std > 0:
                outliers = np.abs(data[col] - mean) > (5 * std)
                outlier_pct = outliers.sum() / len(data)
                if outlier_pct > 0.01:  # > 1% outliers
                    outlier_counts[col] = outlier_pct

        if outlier_counts:
            issues.append(f"High outlier rates: {outlier_counts}")

        # 5. Data recency
        if 'timestamp' in data.columns or isinstance(data.index, pd.DatetimeIndex):
            if isinstance(data.index, pd.DatetimeIndex):
                last_date = data.index.max()
            else:
                last_date = pd.to_datetime(data['timestamp']).max()

            days_since_update = (datetime.now() - last_date).days

            if days_since_update > 7:
                issues.append(f"Data is {days_since_update} days old")

        # Calculate overall quality score
        quality_score = 1.0
        quality_score -= len(missing_cols) * 0.1
        quality_score -= min(missing_pct.mean(), 0.3)
        quality_score -= len(constant_features) * 0.05
        quality_score = max(0.0, quality_score)

        return {
            'quality_score': quality_score,
            'issues': issues,
            'missing_pct': missing_pct.mean(),
            'constant_features': len(constant_features),
            'has_issues': len(issues) > 0
        }


class AutomatedRetrainingSystem:
    """
    Automated model retraining based on performance and drift
    """

    def __init__(self, retraining_config: Dict):
        """
        Initialize retraining system

        Args:
            retraining_config: Configuration with:
                - min_accuracy: Minimum accuracy threshold
                - max_drift: Maximum acceptable drift
                - min_days_between_retrains: Minimum days between retraining
                - data_window_days: Days of data to use for retraining
        """
        self.config = retraining_config
        self.last_retrain_date = None
        self.retrain_history = []

    def should_retrain(self, health_report: ModelHealthReport,
                       drift_summary: Dict,
                       data_quality: Dict) -> Tuple[bool, str]:
        """
        Determine if model should be retrained

        Args:
            health_report: Current model health
            drift_summary: Feature drift analysis
            data_quality: Data quality metrics

        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        reasons = []

        # 1. Check accuracy threshold
        if health_report.current_accuracy < self.config.get('min_accuracy', 0.45):
            reasons.append(
                f"Accuracy below threshold: {health_report.current_accuracy:.1%} "
                f"< {self.config.get('min_accuracy', 0.45):.1%}"
            )

        # 2. Check significant degradation
        if health_report.accuracy_degradation > 0.15:
            reasons.append(
                f"Significant accuracy degradation: {health_report.accuracy_degradation:.1%}"
            )

        # 3. Check feature drift
        if drift_summary['overall_drift_score'] > self.config.get('max_drift', 0.15):
            reasons.append(
                f"High feature drift: {drift_summary['overall_drift_score']:.2f}"
            )

        # 4. Check concept drift
        if health_report.concept_drift_detected:
            reasons.append("Concept drift detected (significant performance change)")

        # 5. Scheduled retraining
        if self.last_retrain_date:
            days_since = (datetime.now() - self.last_retrain_date).days
            scheduled_days = self.config.get('scheduled_retrain_days', 30)

            if days_since >= scheduled_days:
                reasons.append(f"Scheduled retraining ({days_since} days since last retrain)")

        # 6. Check minimum time between retrains
        if reasons and self.last_retrain_date:
            days_since = (datetime.now() - self.last_retrain_date).days
            min_days = self.config.get('min_days_between_retrains', 7)

            if days_since < min_days:
                return False, f"Too soon to retrain (only {days_since} days since last retrain)"

        # 7. Check data quality
        if data_quality.get('quality_score', 1.0) < 0.7:
            return False, "Data quality too low for retraining"

        if reasons:
            return True, "; ".join(reasons)

        return False, "Model performing well, no retraining needed"

    def execute_retraining(self, historical_data: pd.DataFrame,
                           model_pipeline) -> Dict:
        """
        Execute model retraining

        Args:
            historical_data: Data for retraining
            model_pipeline: ML pipeline to retrain

        Returns:
            Dictionary with retraining results
        """
        logger.info("Starting automated model retraining...")

        try:
            # Use recent data window
            days = self.config.get('data_window_days', 180)
            cutoff_date = datetime.now() - timedelta(days=days)

            if isinstance(historical_data.index, pd.DatetimeIndex):
                recent_data = historical_data[historical_data.index >= cutoff_date]
            else:
                recent_data = historical_data.tail(days * 24)  # Approximate for hourly data

            # Retrain model
            results = model_pipeline.train_model(
                model_type=self.config.get('model_type', 'xgboost'),
                hyperparameter_tuning=self.config.get('hyperparameter_tuning', False),
                cross_validation=True
            )

            # Update last retrain date
            self.last_retrain_date = datetime.now()

            # Store in history
            self.retrain_history.append({
                'date': self.last_retrain_date,
                'data_samples': len(recent_data),
                'test_accuracy': results['test_metrics']['accuracy'],
                'test_sharpe': results.get('test_sharpe', 0.0),
                'model_file': results['model_filename']
            })

            logger.info(f"✅ Retraining complete. New accuracy: {results['test_metrics']['accuracy']:.2%}")

            return {
                'success': True,
                'new_accuracy': results['test_metrics']['accuracy'],
                'model_file': results['model_filename'],
                'samples_used': len(recent_data)
            }

        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class ContinuousMonitoringSystem:
    """
    Complete continuous monitoring system that ties everything together
    """

    def __init__(self, model_name: str, baseline_metrics: Dict,
                 baseline_features: pd.DataFrame, retraining_config: Dict):
        """
        Initialize complete monitoring system
        """
        self.accuracy_monitor = ModelAccuracyMonitor(model_name, baseline_metrics)
        self.drift_detector = FeatureDriftDetector(baseline_features)
        self.retraining_system = AutomatedRetrainingSystem(retraining_config)

        self.monitoring_history = []

    def perform_health_check(self,
                             true_labels: np.ndarray,
                             predictions: np.ndarray,
                             current_features: pd.DataFrame,
                             prediction_probas: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive health check

        Args:
            true_labels: Actual outcomes
            predictions: Model predictions
            current_features: Recent feature data
            prediction_probas: Prediction probabilities

        Returns:
            Complete health assessment with recommendations
        """
        logger.info("Performing comprehensive model health check...")

        # 1. Accuracy monitoring
        health_report = self.accuracy_monitor.check_accuracy(
            true_labels, predictions, prediction_probas
        )

        # 2. Feature drift detection
        drift_scores = self.drift_detector.detect_drift(current_features)
        drift_summary = self.drift_detector.get_drift_summary(drift_scores)

        # 3. Data quality check
        expected_cols = list(self.drift_detector.feature_stats.keys())
        data_quality = DataQualityMonitor.check_data_quality(
            current_features, expected_cols
        )

        # 4. Retraining recommendation
        should_retrain, retrain_reason = self.retraining_system.should_retrain(
            health_report, drift_summary, data_quality
        )

        # Compile complete report
        complete_report = {
            'timestamp': datetime.now().isoformat(),
            'model_health': {
                'accuracy': health_report.current_accuracy,
                'baseline_accuracy': health_report.baseline_accuracy,
                'degradation': health_report.accuracy_degradation,
                'confidence_level': health_report.confidence_level,
                'warnings': health_report.warnings
            },
            'drift_analysis': drift_summary,
            'data_quality': data_quality,
            'retraining': {
                'recommended': should_retrain,
                'reason': retrain_reason,
                'last_retrain': self.retraining_system.last_retrain_date
            },
            'overall_status': self._determine_overall_status(
                health_report, drift_summary, data_quality
            )
        }

        # Store in history
        self.monitoring_history.append(complete_report)

        # Log summary
        self._log_summary(complete_report)

        return complete_report

    def _determine_overall_status(self, health_report: ModelHealthReport,
                                  drift_summary: Dict, data_quality: Dict) -> str:
        """Determine overall system status"""

        if (health_report.accuracy_degradation > 0.15 or
            drift_summary['overall_drift_score'] > 0.3 or
                data_quality['quality_score'] < 0.5):
            return 'CRITICAL'

        elif (health_report.accuracy_degradation > 0.10 or
              drift_summary['overall_drift_score'] > 0.2 or
              data_quality['quality_score'] < 0.7):
            return 'WARNING'

        elif health_report.accuracy_degradation > 0.05:
            return 'MONITOR'

        else:
            return 'HEALTHY'

    def _log_summary(self, report: Dict):
        """Log monitoring summary"""
        status = report['overall_status']

        emoji = {
            'HEALTHY': '✅',
            'MONITOR': '👀',
            'WARNING': '⚠️',
            'CRITICAL': '🚨'
        }

        logger.info("=" * 60)
        logger.info(f"{emoji[status]} MODEL HEALTH STATUS: {status}")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {report['model_health']['accuracy']:.2%}")
        logger.info(f"Degradation: {report['model_health']['degradation']:.1%}")
        logger.info(f"Drift Score: {report['drift_analysis']['overall_drift_score']:.2f}")
        logger.info(f"Data Quality: {report['data_quality']['quality_score']:.2f}")

        if report['retraining']['recommended']:
            logger.warning(f"⚠️ RETRAINING RECOMMENDED: {report['retraining']['reason']}")

        logger.info("=" * 60)


# Example usage
if __name__ == "__main__":
    # Simulate monitoring system

    # 1. Set up baseline from training
    baseline_metrics = {
        'accuracy': 0.57,
        'f1_score': 0.59,
        'precision': 0.56,
        'recall': 0.58
    }

    # Baseline features (from training data)
    np.random.seed(42)
    baseline_features = pd.DataFrame({
        'rsi_14': np.random.normal(50, 15, 1000),
        'macd_hist': np.random.normal(0, 2, 1000),
        'volatility_20': np.random.normal(0.02, 0.01, 1000),
        'ema_12': np.random.normal(1.1, 0.05, 1000)
    })

    # Retraining configuration
    retraining_config = {
        'min_accuracy': 0.50,
        'max_drift': 0.15,
        'min_days_between_retrains': 7,
        'data_window_days': 180,
        'model_type': 'xgboost',
        'scheduled_retrain_days': 30
    }

    # 2. Initialize monitoring system
    monitor = ContinuousMonitoringSystem(
        model_name='EUR_USD_Model_v1',
        baseline_metrics=baseline_metrics,
        baseline_features=baseline_features,
        retraining_config=retraining_config
    )

    # 3. Simulate predictions over time
    # First 100 predictions: Good performance
    true_labels_good = np.random.binomial(1, 0.55, 100)  # 55% win rate
    predictions_good = np.random.binomial(1, 0.57, 100)  # Slight edge

    # Simulate feature drift (features shifting)
    current_features_good = pd.DataFrame({
        'rsi_14': np.random.normal(52, 15, 100),  # Slight shift
        'macd_hist': np.random.normal(0.5, 2, 100),
        'volatility_20': np.random.normal(0.021, 0.01, 100),
        'ema_12': np.random.normal(1.11, 0.05, 100)
    })

    print("\n📊 SCENARIO 1: Good Performance")
    report1 = monitor.perform_health_check(
        true_labels_good,
        predictions_good,
        current_features_good
    )

    # Next 100 predictions: Performance degrading
    true_labels_bad = np.random.binomial(1, 0.52, 100)
    predictions_bad = np.random.binomial(1, 0.48, 100)  # Now worse

    # Significant feature drift
    current_features_bad = pd.DataFrame({
        'rsi_14': np.random.normal(60, 20, 100),  # Big shift
        'macd_hist': np.random.normal(1.5, 3, 100),  # Big shift
        'volatility_20': np.random.normal(0.035, 0.015, 100),  # Higher volatility
        'ema_12': np.random.normal(1.15, 0.07, 100)
    })

    print("\n📊 SCENARIO 2: Performance Degrading")
    report2 = monitor.perform_health_check(
        true_labels_bad,
        predictions_bad,
        current_features_bad
    )

    print("\n" + "=" * 60)
    print("MONITORING SUMMARY")
    print("=" * 60)
    print(f"Total checks performed: {len(monitor.monitoring_history)}")
    print(f"Status transitions: HEALTHY → {report2['overall_status']}")
    print(f"Retraining recommended: {report2['retraining']['recommended']}")
