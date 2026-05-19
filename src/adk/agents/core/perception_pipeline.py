"""
Perception Pipeline

Coordinates Loop A (Signal Normalization) and Loop B (State Estimation)
to process user signals into cognitive state estimates.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...utils.schemas import UserSignal, CognitiveState, SignalType
from ...utils.logger import get_logger
from ..loop_a.signal_normalizer import SignalNormalizer
from ..loop_b.state_estimator import StateEstimator


class PerceptionPipeline:
    """
    Perception Pipeline combining Loops A and B

    Flow: Raw Signals -> Normalized Signals -> Cognitive State
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PerceptionPipeline"""
        self.config = config or {}
        self.logger = get_logger("system")

        # Initialize Loop A and B agents
        self.signal_normalizer = SignalNormalizer(config)
        self.state_estimator = StateEstimator(config)

        self.logger.info("PerceptionPipeline initialized")

    async def initialize(self):
        """Initialize the pipeline"""
        await self.state_estimator.initialize()
        self.logger.info("PerceptionPipeline ready")

    async def process_signals(
        self,
        raw_signals: List[tuple[SignalType, Any, Optional[Dict[str, Any]]]],
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[List[UserSignal], CognitiveState]:
        """
        Process raw signals through the perception pipeline

        Args:
            raw_signals: List of (signal_type, raw_value, metadata) tuples
            context: Optional contextual information

        Returns:
            Tuple of (normalized_signals, cognitive_state)
        """
        # Loop A: Normalize signals
        normalized_signals = await self.signal_normalizer.normalize_batch(raw_signals)

        # Loop B: Estimate cognitive state
        cognitive_state = await self.state_estimator.estimate_state(
            normalized_signals,
            context
        )

        return normalized_signals, cognitive_state

    async def close(self):
        """Clean up resources"""
        await self.state_estimator.close()
