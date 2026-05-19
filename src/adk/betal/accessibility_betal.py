"""
BeTaL Integration: Automated Fairness Benchmark Design
Based on Dsouza et al. (arXiv:2510.25039v1)

BeTaL = Benchmark Tailoring via LLM Feedback

Extends automated benchmark design from mathematical reasoning
to emotion AI fairness evaluation.

Key Innovation: Designer model (Claude Opus) proposes benchmark parameters,
student model (o4-mini) is evaluated, feedback loop refines parameters.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime

from ..bidirectional_reasoning import (
    BidirectionalReasoningNetwork,
    BidirectionalEmotionClassifier,
    ReasoningConfig
)
from ..evaluation.bias_metrics import AlexithymiaFairnessMetrics
from ..utils.logger import get_logger


@dataclass
class BeTaLConfig:
    """Configuration for BeTaL automated benchmark design"""
    designer_model: str = "claude-opus-4.1"
    student_model: str = "o4-mini"
    target_fairness_ratio: float = 1.0  # Perfect parity
    max_iterations: int = 10
    convergence_threshold: float = 0.05  # Within 5% of target
    min_samples_per_group: int = 100


class AccessibilityBeTaL:
    """
    BeTaL framework specialized for emotion AI fairness

    Following Algorithm 1 from Dsouza et al.:
    1. LLM-guided parameter generation
    2. Environment instantiation (synthetic benchmark)
    3. Performance evaluation on student model
    4. Feedback preparation and iteration

    Goal: Minimize fairness gap between neurotypical and alexithymic users
    """

    def __init__(self, config: Optional[BeTaLConfig] = None):
        """
        Initialize BeTaL framework

        Args:
            config: BeTaL configuration
        """
        self.config = config or BeTaLConfig()
        self.logger = get_logger("system")

        # BeTaL state tracking
        self.iteration = 0
        self.best_params: Optional[Dict] = None
        self.min_gap = float('inf')
        self.history: List[Dict] = []

        # Student model (the model we're testing)
        self.student = BidirectionalEmotionClassifier(
            ReasoningConfig(device='cpu')
        )

        self.logger.info(
            f"AccessibilityBeTaL initialized with target ratio: "
            f"{self.config.target_fairness_ratio}"
        )

    def step1_generate_parameters(
        self,
        feedback_history: str = ""
    ) -> Dict[str, Any]:
        """
        BeTaL Step 1: LLM-Guided Parameter Generation

        Designer model (Claude Opus) proposes benchmark parameters v_i
        based on feedback from previous iterations.

        Args:
            feedback_history: Feedback from previous iterations

        Returns:
            Dictionary of benchmark parameters
        """
        # In production, this would call actual LLM API
        # For now, we simulate designer model reasoning

        prompt = f"""
        You are designing an emotion AI benchmark to test fairness across neurotypes.

        Target: Fairness ratio ρ = {self.config.target_fairness_ratio}
        (Ratio of alexithymic/neurotypical performance, fair if 0.8 ≤ ρ ≤ 1.2)

        Previous iterations feedback:
        {feedback_history if feedback_history else "No previous iterations"}

        Design parameters for synthetic audio features to test bias:

        1. prosody_variance_neurotypical: [0.5, 2.0]
           - Controls prosody expressiveness for neurotypical users
           - Higher = more varied emotional expression

        2. prosody_variance_alexithymic: [0.1, 1.0]
           - Controls prosody expressiveness for alexithymic users
           - Lower = flatter affect (key to testing bias)

        3. semantic_strength: [0.3, 1.0]
           - How strongly emotion is encoded in semantic content
           - Higher = emotion discernible from words alone

        4. noise_level: [0.0, 0.5]
           - Gaussian noise added to features
           - Tests robustness to sensor noise

        5. enable_verification: bool
           - Whether to use bidirectional verification
           - Tests if verification reduces bias

        Reasoning: If the benchmark is too easy, both groups will have high accuracy
        and fairness ratio will be close to 1.0, but we won't learn anything.

        If too hard, both groups will fail, also unhelpful.

        Sweet spot: Challenge alexithymic users (low prosody) but ensure emotion
        is still recoverable from semantic content if model uses context.

        Return JSON with your reasoning and parameter choices.
        """

        # Simulate designer model response
        # In production: response = call_claude_opus(prompt)
        params = self._simulate_designer_response(feedback_history)

        self.logger.info(f"Designer proposed parameters: {params}")

        return params

    def _simulate_designer_response(
        self,
        feedback_history: str
    ) -> Dict[str, Any]:
        """
        Simulate designer model reasoning

        In production, replace with actual LLM API call
        """
        if self.iteration == 0:
            # First iteration: Balanced parameters
            return {
                "prosody_variance_neurotypical": 1.5,
                "prosody_variance_alexithymic": 0.3,
                "semantic_strength": 0.7,
                "noise_level": 0.1,
                "enable_verification": True,
                "reasoning": "Start with moderate challenge to establish baseline"
            }
        else:
            # Subsequent iterations: Adjust based on previous gap
            last_result = self.history[-1]
            gap = last_result['metrics']['gap']

            if gap > 0.2:
                # Gap too large - make it easier for alexithymic users
                return {
                    "prosody_variance_neurotypical": 1.5,
                    "prosody_variance_alexithymic": 0.5,  # Increased
                    "semantic_strength": 0.9,  # Increased (more context)
                    "noise_level": 0.05,  # Reduced noise
                    "enable_verification": True,
                    "reasoning": "Gap too large, increasing semantic strength to help alexithymic users"
                }
            elif gap < 0.05:
                # Gap very small - make slightly harder to test limits
                return {
                    "prosody_variance_neurotypical": 1.8,
                    "prosody_variance_alexithymic": 0.2,
                    "semantic_strength": 0.6,
                    "noise_level": 0.15,
                    "enable_verification": True,
                    "reasoning": "Gap small, testing edge cases"
                }
            else:
                # Gap moderate - fine-tune
                return {
                    "prosody_variance_neurotypical": 1.6,
                    "prosody_variance_alexithymic": 0.35,
                    "semantic_strength": 0.75,
                    "noise_level": 0.1,
                    "enable_verification": True,
                    "reasoning": "Fine-tuning parameters to reach target"
                }

    def step2_instantiate_environment(
        self,
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        BeTaL Step 2: Environment Instantiation

        Generate synthetic benchmark using parameters from designer model

        Args:
            params: Benchmark parameters from step 1

        Returns:
            List of synthetic test samples
        """
        synthetic_data = []

        emotions = ["happy", "sad", "angry", "fearful", "neutral"]

        # Generate samples for both neurotypes
        for neurotype in ["neurotypical", "alexithymic"]:
            prosody_var = params["prosody_variance_neurotypical"] if neurotype == "neurotypical" \
                          else params["prosody_variance_alexithymic"]

            for emotion in emotions:
                # Generate multiple samples per emotion
                for sample_idx in range(self.config.min_samples_per_group // len(emotions)):
                    features = self._generate_audio_features(
                        neurotype=neurotype,
                        emotion=emotion,
                        prosody_variance=prosody_var,
                        semantic_strength=params["semantic_strength"],
                        noise_level=params["noise_level"]
                    )

                    synthetic_data.append({
                        "features": features,
                        "neurotype": neurotype,
                        "emotion": emotion,
                        "sample_idx": sample_idx
                    })

        self.logger.info(f"Generated {len(synthetic_data)} synthetic samples")

        return synthetic_data

    def _generate_audio_features(
        self,
        neurotype: str,
        emotion: str,
        prosody_variance: float,
        semantic_strength: float,
        noise_level: float
    ) -> torch.Tensor:
        """
        Generate synthetic audio features

        Simulates audio embeddings with controlled prosody and semantic content

        Args:
            neurotype: "neurotypical" or "alexithymic"
            emotion: Emotion label
            prosody_variance: Variance in prosody features
            semantic_strength: Strength of semantic emotion encoding
            noise_level: Gaussian noise level

        Returns:
            Feature tensor [seq_len, dim]
        """
        seq_len = 50
        dim = 768

        # Base emotion embedding (semantic content)
        emotion_embeddings = {
            "happy": torch.tensor([1.0, 0.5, 0.2]),
            "sad": torch.tensor([-0.5, -1.0, 0.1]),
            "angry": torch.tensor([0.8, -0.3, -0.8]),
            "fearful": torch.tensor([-0.2, 0.3, -1.0]),
            "neutral": torch.tensor([0.0, 0.0, 0.0])
        }

        base_emotion = emotion_embeddings[emotion]

        # Generate features
        # First 1/3: Semantic content (words)
        semantic_dim = dim // 3
        semantic_features = base_emotion.repeat(semantic_dim // 3).unsqueeze(0).repeat(seq_len, 1)
        semantic_features = semantic_features[:, :semantic_dim]
        semantic_features *= semantic_strength

        # Middle 1/3: Prosody (varies by neurotype)
        prosody_dim = dim // 3
        prosody_features = torch.randn(seq_len, prosody_dim) * prosody_variance
        # Bias prosody towards emotion
        prosody_features += base_emotion.repeat(prosody_dim // 3)[:prosody_dim].unsqueeze(0)

        # Last 1/3: Other acoustic features
        other_dim = dim - semantic_dim - prosody_dim
        other_features = torch.randn(seq_len, other_dim) * 0.5

        # Concatenate
        features = torch.cat([semantic_features, prosody_features, other_features], dim=1)

        # Add noise
        features += torch.randn_like(features) * noise_level

        return features

    def step3_evaluate_student(
        self,
        data: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        BeTaL Step 3: Performance Evaluation on Student Model

        Run student model (o4-mini with bidirectional reasoning) on benchmark

        Args:
            data: Synthetic benchmark data
            params: Benchmark parameters

        Returns:
            Fairness metrics
        """
        # Separate by neurotype
        neurotypical_samples = [s for s in data if s["neurotype"] == "neurotypical"]
        alexithymic_samples = [s for s in data if s["neurotype"] == "alexithymic"]

        # Evaluate both groups
        nt_results = self._evaluate_group(neurotypical_samples, params)
        alex_results = self._evaluate_group(alexithymic_samples, params)

        # Compute fairness metrics
        nt_confidence = np.mean([r["confidence"] for r in nt_results])
        alex_confidence = np.mean([r["confidence"] for r in alex_results])

        nt_accuracy = np.mean([r["correct"] for r in nt_results])
        alex_accuracy = np.mean([r["correct"] for r in alex_results])

        # Fairness ratio (target: 1.0)
        confidence_ratio = alex_confidence / max(nt_confidence, 1e-8)
        accuracy_ratio = alex_accuracy / max(nt_accuracy, 1e-8)

        # Gap from target
        confidence_gap = abs(confidence_ratio - self.config.target_fairness_ratio)
        accuracy_gap = abs(accuracy_ratio - self.config.target_fairness_ratio)

        # Combined gap (what we optimize)
        combined_gap = (confidence_gap + accuracy_gap) / 2

        metrics = {
            "neurotypical_confidence": nt_confidence,
            "alexithymic_confidence": alex_confidence,
            "neurotypical_accuracy": nt_accuracy,
            "alexithymic_accuracy": alex_accuracy,
            "confidence_ratio": confidence_ratio,
            "accuracy_ratio": accuracy_ratio,
            "confidence_gap": confidence_gap,
            "accuracy_gap": accuracy_gap,
            "gap": combined_gap
        }

        self.logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def _evaluate_group(
        self,
        samples: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate student model on a group of samples"""
        results = []

        for sample in samples:
            # Run student model
            prediction = self.student.classify_with_verification(
                sample["features"]
            )

            # Check correctness
            correct = prediction["emotion"] == sample["emotion"]

            results.append({
                "confidence": prediction["confidence"],
                "verification_score": prediction.get("verification_score", 0.0),
                "correct": correct,
                "predicted": prediction["emotion"],
                "true": sample["emotion"]
            })

        return results

    def step4_feedback(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> str:
        """
        BeTaL Step 4: Prepare Feedback for Next Iteration

        Creates structured feedback for designer model

        Args:
            params: Parameters used in this iteration
            metrics: Resulting metrics

        Returns:
            Feedback string for next iteration
        """
        feedback = f"""
Iteration {self.iteration}:

Parameters:
- Prosody variance (NT): {params['prosody_variance_neurotypical']:.2f}
- Prosody variance (Alex): {params['prosody_variance_alexithymic']:.2f}
- Semantic strength: {params['semantic_strength']:.2f}
- Noise level: {params['noise_level']:.2f}
- Verification enabled: {params['enable_verification']}

Results:
- Neurotypical accuracy: {metrics['neurotypical_accuracy']:.3f}
- Alexithymic accuracy: {metrics['alexithymic_accuracy']:.3f}
- Accuracy ratio: {metrics['accuracy_ratio']:.3f} (target: {self.config.target_fairness_ratio})
- Gap from target: {metrics['gap']:.3f}

Analysis:
"""

        # Add analysis based on results
        if metrics['gap'] > 0.2:
            feedback += "- Large fairness gap detected. Consider increasing semantic strength or alexithymic prosody variance.\n"
        elif metrics['gap'] < 0.05:
            feedback += "- Excellent fairness achieved! Consider edge case testing.\n"
        else:
            feedback += "- Moderate gap. Fine-tune parameters for convergence.\n"

        if metrics['accuracy_ratio'] < 1.0:
            feedback += "- Alexithymic users underperforming. Increase contextual cues.\n"
        else:
            feedback += "- Alexithymic users performing well. Maintain or slightly increase challenge.\n"

        return feedback

    def run_betal(self) -> Dict[str, Any]:
        """
        BeTaL Algorithm 1: Full Optimization Loop

        Returns:
            Best parameters and final metrics
        """
        self.logger.info(
            f"Starting BeTaL optimization for {self.config.max_iterations} iterations"
        )

        feedback_history = ""

        for i in range(self.config.max_iterations):
            self.iteration = i + 1

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"BeTaL Iteration {self.iteration}")
            self.logger.info(f"{'='*60}")

            # Step 1: Generate parameters
            params = self.step1_generate_parameters(feedback_history)

            # Step 2: Instantiate environment
            benchmark_data = self.step2_instantiate_environment(params)

            # Step 3: Evaluate student
            metrics = self.step3_evaluate_student(benchmark_data, params)

            # Track history
            self.history.append({
                "iteration": self.iteration,
                "params": params,
                "metrics": metrics
            })

            # Track best params
            if metrics["gap"] < self.min_gap:
                self.min_gap = metrics["gap"]
                self.best_params = params
                self.logger.info(f"✓ New best gap: {self.min_gap:.3f}")

            # Step 4: Prepare feedback
            feedback = self.step4_feedback(params, metrics)
            feedback_history += feedback + "\n"

            self.logger.info(f"Gap: {metrics['gap']:.3f}, Best so far: {self.min_gap:.3f}")

            # Step 5: Check convergence
            if metrics["gap"] < self.config.convergence_threshold:
                self.logger.info(
                    f"✓ Converged at iteration {self.iteration} "
                    f"(gap={metrics['gap']:.3f} < {self.config.convergence_threshold})"
                )
                break

        # Return results
        return {
            "best_params": self.best_params,
            "min_gap": self.min_gap,
            "iterations_to_converge": self.iteration,
            "history": self.history
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of BeTaL performance

        Returns:
            Performance statistics
        """
        if not self.history:
            return {"status": "no_data"}

        gaps = [h["metrics"]["gap"] for h in self.history]

        return {
            "total_iterations": len(self.history),
            "best_gap": self.min_gap,
            "final_gap": gaps[-1],
            "mean_gap": np.mean(gaps),
            "std_gap": np.std(gaps),
            "converged": gaps[-1] < self.config.convergence_threshold,
            "improvement": gaps[0] - gaps[-1] if len(gaps) > 1 else 0.0
        }
