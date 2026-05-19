"""
Bias evaluation metrics for neuroadaptive accessibility

Specifically designed to measure fairness for neurodivergent users,
particularly those with alexithymia.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class AlexithymiaFairnessMetrics:
    """
    Metrics for evaluating alexithymia fairness in emotion AI

    Key metrics:
    - Verification Rate Parity: Neurotypical vs. Alexithymic users
    - False Negative Rate: Missed emotions due to flat affect
    - Confidence Calibration: Are low-verification scores appropriate?
    """

    def __init__(self):
        """Initialize metrics tracker"""
        self.results = {
            'neurotypical': [],
            'alexithymic': []
        }

    def add_prediction(
        self,
        prediction: Dict,
        ground_truth: str,
        alexithymia_score: float
    ):
        """
        Add a prediction for evaluation

        Args:
            prediction: Dict with 'emotion', 'confidence', 'is_verified'
            ground_truth: True emotion label
            alexithymia_score: User's alexithymia score (0-1)
        """
        group = 'alexithymic' if alexithymia_score > 0.5 else 'neurotypical'

        result = {
            'predicted': prediction['emotion'],
            'true': ground_truth,
            'confidence': prediction['confidence'],
            'verified': prediction['is_verified'],
            'correct': prediction['emotion'] == ground_truth,
            'alexithymia_score': alexithymia_score
        }

        self.results[group].append(result)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive fairness metrics

        Returns:
            Dictionary of fairness metrics
        """
        metrics = {}

        for group in ['neurotypical', 'alexithymic']:
            if not self.results[group]:
                continue

            results = self.results[group]

            # Accuracy
            accuracy = np.mean([r['correct'] for r in results])
            metrics[f'{group}_accuracy'] = accuracy

            # Verification rate
            verification_rate = np.mean([r['verified'] for r in results])
            metrics[f'{group}_verification_rate'] = verification_rate

            # False negative rate (missed emotions)
            false_negatives = sum(1 for r in results if not r['correct'] and r['true'] != 'neutral')
            total_emotional = sum(1 for r in results if r['true'] != 'neutral')
            fnr = false_negatives / max(1, total_emotional)
            metrics[f'{group}_false_negative_rate'] = fnr

            # Average confidence
            avg_confidence = np.mean([r['confidence'] for r in results])
            metrics[f'{group}_avg_confidence'] = avg_confidence

        # Fairness metrics (parity between groups)
        if self.results['neurotypical'] and self.results['alexithymic']:
            # Verification Rate Parity
            verification_parity = abs(
                metrics['neurotypical_verification_rate'] -
                metrics['alexithymic_verification_rate']
            )
            metrics['verification_parity_gap'] = verification_parity

            # Accuracy Parity
            accuracy_parity = abs(
                metrics['neurotypical_accuracy'] -
                metrics['alexithymic_accuracy']
            )
            metrics['accuracy_parity_gap'] = accuracy_parity

            # False Negative Parity
            fnr_parity = abs(
                metrics['neurotypical_false_negative_rate'] -
                metrics['alexithymic_false_negative_rate']
            )
            metrics['fnr_parity_gap'] = fnr_parity

            # Overall fairness score (lower is better, 0 = perfect parity)
            metrics['overall_fairness_score'] = (
                verification_parity * 0.4 +
                accuracy_parity * 0.4 +
                fnr_parity * 0.2
            )

        return metrics

    def print_report(self):
        """Print detailed fairness report"""
        metrics = self.compute_metrics()

        print("=" * 60)
        print("ALEXITHYMIA FAIRNESS EVALUATION REPORT")
        print("=" * 60)

        # Per-group metrics
        for group in ['neurotypical', 'alexithymic']:
            if f'{group}_accuracy' not in metrics:
                continue

            print(f"\n{group.upper()} GROUP:")
            print(f"  Accuracy:           {metrics[f'{group}_accuracy']:.3f}")
            print(f"  Verification Rate:  {metrics[f'{group}_verification_rate']:.3f}")
            print(f"  False Negative Rate: {metrics[f'{group}_false_negative_rate']:.3f}")
            print(f"  Avg Confidence:     {metrics[f'{group}_avg_confidence']:.3f}")

        # Fairness metrics
        if 'overall_fairness_score' in metrics:
            print("\nFAIRNESS METRICS:")
            print(f"  Verification Parity Gap: {metrics['verification_parity_gap']:.3f}")
            print(f"  Accuracy Parity Gap:     {metrics['accuracy_parity_gap']:.3f}")
            print(f"  FNR Parity Gap:          {metrics['fnr_parity_gap']:.3f}")
            print(f"  Overall Fairness Score:  {metrics['overall_fairness_score']:.3f}")

            # Interpretation
            fairness_score = metrics['overall_fairness_score']
            if fairness_score < 0.1:
                interpretation = "EXCELLENT - Near-perfect parity"
            elif fairness_score < 0.2:
                interpretation = "GOOD - Acceptable fairness"
            elif fairness_score < 0.3:
                interpretation = "FAIR - Some bias present"
            else:
                interpretation = "POOR - Significant bias detected"

            print(f"\n  Interpretation: {interpretation}")

        print("=" * 60)


class BidirectionalConsistencyMetrics:
    """
    Metrics for bidirectional consistency

    Measures how well forward and reverse reasoning align
    """

    def __init__(self):
        """Initialize metrics tracker"""
        self.consistency_scores = []

    def add_prediction(
        self,
        forward_output: torch.Tensor,
        reverse_output: torch.Tensor,
        verification_score: float
    ):
        """
        Add prediction for consistency evaluation

        Args:
            forward_output: Forward prediction tensor
            reverse_output: Reverse reconstruction tensor
            verification_score: Bidirectional verification score
        """
        # Compute reconstruction error
        mse = torch.nn.functional.mse_loss(forward_output, reverse_output)

        self.consistency_scores.append({
            'reconstruction_error': mse.item(),
            'verification_score': verification_score,
            'consistent': verification_score > 0.7
        })

    def compute_metrics(self) -> Dict[str, float]:
        """Compute consistency metrics"""
        if not self.consistency_scores:
            return {}

        return {
            'avg_reconstruction_error': np.mean([s['reconstruction_error'] for s in self.consistency_scores]),
            'avg_verification_score': np.mean([s['verification_score'] for s in self.consistency_scores]),
            'consistency_rate': np.mean([s['consistent'] for s in self.consistency_scores])
        }


def evaluate_bias_mitigation(
    model,
    test_data_neurotypical: List[Dict],
    test_data_alexithymic: List[Dict],
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Comprehensive bias mitigation evaluation

    Args:
        model: BidirectionalEmotionClassifier
        test_data_neurotypical: Test data for neurotypical users
        test_data_alexithymic: Test data for alexithymic users (with flat affect)
        device: Device to run on

    Returns:
        Comprehensive metrics dictionary
    """
    fairness_metrics = AlexithymiaFairnessMetrics()
    consistency_metrics = BidirectionalConsistencyMetrics()

    # Evaluate neurotypical users
    for item in test_data_neurotypical:
        audio_features = torch.tensor(item['audio_features'], device=device)
        ground_truth = item['emotion']

        prediction = model.classify_with_verification(audio_features)

        fairness_metrics.add_prediction(
            prediction,
            ground_truth,
            alexithymia_score=0.0
        )

    # Evaluate alexithymic users
    for item in test_data_alexithymic:
        audio_features = torch.tensor(item['audio_features'], device=device)
        ground_truth = item['emotion']

        prediction = model.classify_with_verification(audio_features)

        fairness_metrics.add_prediction(
            prediction,
            ground_truth,
            alexithymia_score=1.0
        )

    # Compute and print metrics
    fairness_results = fairness_metrics.compute_metrics()
    fairness_metrics.print_report()

    return fairness_results
