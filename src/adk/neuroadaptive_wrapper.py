"""
Neuroadaptive Wrapper for DeepAgent ADK

Integrates bidirectional reasoning with the accessibility coordinator
to provide alexithymia-aware emotion recognition and bias mitigation.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch
import numpy as np

from .bidirectional_reasoning import BidirectionalEmotionClassifier, ReasoningConfig
from .agents.core import AccessibilityCoordinator
from .utils import CognitiveState, SignalType, get_logger


class NeuroadaptiveWrapper:
    """
    Neuroadaptive wrapper integrating bidirectional reasoning
    with accessibility coordination

    Key Features:
    - Bidirectional emotion verification
    - Alexithymia-aware adaptations
    - Bias mitigation through contrastive learning
    - Real-time accessibility adjustments
    """

    def __init__(
        self,
        accessibility_coordinator: Optional[AccessibilityCoordinator] = None,
        user_profile: Optional[Dict] = None,
        reasoning_config: Optional[ReasoningConfig] = None
    ):
        """
        Initialize neuroadaptive wrapper

        Args:
            accessibility_coordinator: Existing AccessibilityCoordinator instance
            user_profile: User accessibility profile
            reasoning_config: Configuration for bidirectional reasoning
        """
        self.logger = get_logger("system")

        # Initialize or use existing coordinator
        self.coordinator = accessibility_coordinator or AccessibilityCoordinator()

        # Initialize bidirectional emotion classifier
        self.emotion_classifier = BidirectionalEmotionClassifier(
            reasoning_config or ReasoningConfig(device='cpu')
        )

        # User profile
        self.user_profile = user_profile or {}
        self.alexithymia_score = self.user_profile.get("alexithymia_score", 0.0)
        self.neurodivergent_flags = self.user_profile.get("neurodivergent_flags", [])

        # Tracking
        self.emotion_history: List[Dict] = []
        self.verification_failures: List[Dict] = []
        self.bias_mitigation_stats: Dict[str, int] = {
            "alexithymia_detected": 0,
            "verification_failures": 0,
            "bias_corrections": 0
        }

        self.logger.info(
            f"NeuroadaptiveWrapper initialized "
            f"(alexithymia_score: {self.alexithymia_score:.2f})"
        )

    async def initialize(self):
        """Initialize the wrapper and underlying components"""
        await self.coordinator.initialize()
        self.logger.info("NeuroadaptiveWrapper ready")

    async def process_interaction_with_emotion(
        self,
        raw_signals: List[tuple],
        audio_features: Optional[torch.Tensor] = None,
        text_content: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process user interaction with emotion classification and verification

        Args:
            raw_signals: Raw user signals for accessibility
            audio_features: Optional audio features for emotion detection
            text_content: Optional text content to refine
            user_id: User identifier
            context: Additional context

        Returns:
            Comprehensive result with accessibility + emotion analysis
        """
        start_time = datetime.now()

        # Step 1: Standard accessibility processing
        accessibility_result = await self.coordinator.process_user_interaction(
            raw_signals=raw_signals,
            user_id=user_id,
            content_to_refine=text_content,
            context=context
        )

        # Step 2: Emotion classification with bidirectional verification
        emotion_result = None
        if audio_features is not None:
            emotion_result = await self._classify_emotion_with_bias_mitigation(
                audio_features,
                accessibility_result['cognitive_state']
            )

            # Add emotion to context
            self.emotion_history.append({
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion_result['emotion'],
                'confidence': emotion_result['confidence'],
                'verified': emotion_result['is_verified']
            })

        # Step 3: Compute enhanced accessibility metrics
        duration = (datetime.now() - start_time).total_seconds()
        enhanced_metrics = self._compute_enhanced_metrics(
            accessibility_result,
            emotion_result,
            duration
        )

        # Step 4: Apply alexithymia-aware adaptations
        if self.alexithymia_score > 0.3:
            enhanced_adaptations = self._apply_alexithymia_adaptations(
                accessibility_result['ui_adaptations'],
                emotion_result
            )
        else:
            enhanced_adaptations = accessibility_result['ui_adaptations']

        # Compile complete result
        complete_result = {
            **accessibility_result,
            'emotion_analysis': emotion_result,
            'enhanced_adaptations': enhanced_adaptations,
            'enhanced_metrics': enhanced_metrics,
            'bias_mitigation_stats': self.bias_mitigation_stats.copy()
        }

        return complete_result

    async def _classify_emotion_with_bias_mitigation(
        self,
        audio_features: torch.Tensor,
        cognitive_state: Dict
    ) -> Dict[str, Any]:
        """
        Classify emotion with bidirectional verification and bias mitigation

        Args:
            audio_features: Audio feature tensor
            cognitive_state: Current cognitive state

        Returns:
            Enhanced emotion result with bias mitigation flags
        """
        # Run bidirectional classifier
        emotion_result = self.emotion_classifier.classify_with_verification(
            audio_features
        )

        # Bias mitigation: Check for alexithymia patterns
        if not emotion_result['is_verified'] and self.alexithymia_score > 0.5:
            # This is EXPECTED for alexithymic users - not an error!
            self.bias_mitigation_stats['alexithymia_detected'] += 1

            # Don't penalize low verification score
            emotion_result['alexithymia_indicator'] = 1.0 - emotion_result['verification_score']
            emotion_result['bias_mitigation'] = "alexithymia_aware"

            self.logger.info(
                f"Alexithymia pattern detected (verification: {emotion_result['verification_score']:.2f}). "
                "This is expected and not treated as error."
            )

        elif not emotion_result['is_verified']:
            # Non-alexithymic user with low verification - potential issue
            self.verification_failures.append({
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion_result['emotion'],
                'verification_score': emotion_result['verification_score'],
                'cognitive_state': cognitive_state
            })
            self.bias_mitigation_stats['verification_failures'] += 1

        # Additional context from cognitive state
        if cognitive_state.get('stress_level', 0) > 0.7:
            # High stress might affect emotion expression
            emotion_result['stress_adjusted'] = True
            emotion_result['original_emotion'] = emotion_result['emotion']

            # Bias correction: Don't over-interpret stressed signals
            if emotion_result['emotion'] in ['angry', 'anxious']:
                emotion_result['confidence'] *= 0.8  # Reduce confidence
                self.bias_mitigation_stats['bias_corrections'] += 1

        return emotion_result

    def _compute_enhanced_metrics(
        self,
        accessibility_result: Dict,
        emotion_result: Optional[Dict],
        duration: float
    ) -> Dict[str, float]:
        """
        Compute enhanced accessibility metrics including emotion awareness

        Args:
            accessibility_result: Standard accessibility result
            emotion_result: Emotion classification result
            duration: Processing duration

        Returns:
            Enhanced metrics dictionary
        """
        base_metrics = accessibility_result.get('metrics', {})

        enhanced = {
            **base_metrics,
            'processing_duration_ms': duration * 1000,
        }

        # Add emotion-specific metrics
        if emotion_result:
            enhanced['emotion_confidence'] = emotion_result['confidence']
            enhanced['emotion_verification_score'] = emotion_result['verification_score']

            # Alexithymia fairness metric
            if 'alexithymia_indicator' in emotion_result:
                enhanced['alexithymia_fairness_score'] = 1.0 - emotion_result['alexithymia_indicator']
            else:
                enhanced['alexithymia_fairness_score'] = 1.0

            # Overall bias mitigation score
            total_interactions = len(self.emotion_history)
            if total_interactions > 0:
                bias_correction_rate = self.bias_mitigation_stats['bias_corrections'] / total_interactions
                enhanced['bias_mitigation_score'] = 1.0 - bias_correction_rate
            else:
                enhanced['bias_mitigation_score'] = 1.0

        return enhanced

    def _apply_alexithymia_adaptations(
        self,
        base_adaptations: List[Dict],
        emotion_result: Optional[Dict]
    ) -> List[Dict]:
        """
        Apply alexithymia-specific UI adaptations

        Args:
            base_adaptations: Standard UI adaptations
            emotion_result: Emotion classification result

        Returns:
            Enhanced adaptations list
        """
        enhanced = base_adaptations.copy()

        # Alexithymia-specific adaptations
        if self.alexithymia_score > 0.5:
            # 1. Increase explicit emotion labeling
            enhanced.append({
                'category': 'emotion_labeling',
                'parameter': 'enable_explicit_labels',
                'value': True,
                'rationale': 'Explicit emotion labels for alexithymic users',
                'priority': 9
            })

            # 2. Reduce reliance on prosody-based feedback
            enhanced.append({
                'category': 'audio_feedback',
                'parameter': 'reduce_prosody_reliance',
                'value': 0.5,
                'rationale': 'Alexithymic users may have flat affect',
                'priority': 8
            })

            # 3. Provide alternative emotion expression channels
            enhanced.append({
                'category': 'input_modality',
                'parameter': 'enable_emoji_selector',
                'value': True,
                'rationale': 'Alternative emotion expression for alexithymia',
                'priority': 7
            })

        # If verification failed but user is NOT alexithymic, different approach
        if emotion_result and not emotion_result.get('is_verified') and self.alexithymia_score < 0.3:
            enhanced.append({
                'category': 'emotion_clarification',
                'parameter': 'request_explicit_feedback',
                'value': True,
                'rationale': 'Verification failed - request user clarification',
                'priority': 9
            })

        return enhanced

    async def get_bias_mitigation_report(self) -> Dict[str, Any]:
        """
        Generate bias mitigation report

        Returns:
            Comprehensive bias mitigation statistics
        """
        total_interactions = len(self.emotion_history)

        if total_interactions == 0:
            return {
                'status': 'no_data',
                'message': 'No interactions processed yet'
            }

        # Calculate metrics
        verified_count = sum(1 for e in self.emotion_history if e['verified'])
        verification_rate = verified_count / total_interactions

        avg_confidence = np.mean([e['confidence'] for e in self.emotion_history])

        # Alexithymia fairness: How often did we correctly handle alexithymia?
        alexithymia_fairness = 1.0 - (
            self.bias_mitigation_stats['verification_failures'] /
            max(1, total_interactions)
        )

        return {
            'status': 'ok',
            'total_interactions': total_interactions,
            'verification_rate': verification_rate,
            'avg_confidence': avg_confidence,
            'alexithymia_fairness_score': alexithymia_fairness,
            'bias_mitigation_stats': self.bias_mitigation_stats.copy(),
            'verification_failures': len(self.verification_failures),
            'user_profile': {
                'alexithymia_score': self.alexithymia_score,
                'neurodivergent_flags': self.neurodivergent_flags
            }
        }

    async def close(self):
        """Clean up resources"""
        await self.coordinator.close()
