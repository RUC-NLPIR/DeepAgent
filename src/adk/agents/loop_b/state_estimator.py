"""
StateEstimator Agent - Loop B

This agent estimates the user's cognitive state based on normalized signals
from Loop A, optionally using the XGC-AVis service for advanced estimation.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque

from ...utils.schemas import UserSignal, CognitiveState, SignalType
from ...utils.config_loader import get_config_value
from ...utils.logger import get_logger
from .xgc_avis_client import XGCAVisClient


class StateEstimator:
    """
    Agent for estimating user cognitive state

    This agent processes normalized signals and estimates various cognitive
    dimensions including cognitive load, attention, fatigue, stress, and
    reading comprehension.

    It can use either:
    1. Built-in heuristic estimation
    2. External XGC-AVis service for advanced ML-based estimation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the StateEstimator agent

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger("system")

        # Load configuration
        self.enabled = get_config_value("loop_b.enabled", True)
        self.state_dimensions = get_config_value(
            "loop_b.state_dimensions",
            ["cognitive_load", "attention_level", "fatigue_index",
             "stress_level", "reading_comprehension"]
        )
        self.update_frequency_ms = get_config_value(
            "loop_b.update_frequency_ms",
            500
        )

        # XGC-AVis client
        self.xgc_client = XGCAVisClient(config)
        self.use_xgc_avis = False  # Will be set after health check

        # State tracking
        self.current_state: Optional[CognitiveState] = None
        self.state_history: deque = deque(maxlen=100)
        self.signal_buffer: deque = deque(maxlen=50)
        self.last_update_time: Optional[datetime] = None

        self.logger.info("StateEstimator initialized")

    async def initialize(self):
        """Initialize the estimator and check XGC-AVis availability"""
        # Check if XGC-AVis service is available
        async with self.xgc_client as client:
            self.use_xgc_avis = await client.health_check()

        if self.use_xgc_avis:
            self.logger.info("XGC-AVis service is available and will be used")
        else:
            self.logger.info("XGC-AVis service unavailable. Using heuristic estimation")

    async def estimate_state(
        self,
        signals: List[UserSignal],
        context: Optional[Dict[str, Any]] = None
    ) -> CognitiveState:
        """
        Estimate cognitive state from user signals

        Args:
            signals: List of normalized user signals
            context: Optional contextual information

        Returns:
            Estimated CognitiveState
        """
        if not self.enabled:
            self.logger.warning("StateEstimator is disabled")
            return self._get_default_state()

        # Add signals to buffer
        self.signal_buffer.extend(signals)

        # Check if enough time has passed for an update
        if self.last_update_time is not None:
            time_since_update = (datetime.now() - self.last_update_time).total_seconds() * 1000
            if time_since_update < self.update_frequency_ms:
                # Return cached state if too soon
                if self.current_state:
                    return self.current_state

        # Estimate state
        if self.use_xgc_avis:
            state = await self._estimate_with_xgc_avis(signals, context)
        else:
            state = await self._estimate_heuristic(signals, context)

        # Update tracking
        self.current_state = state
        self.state_history.append(state)
        self.last_update_time = datetime.now()

        self.logger.debug(
            f"Estimated state - Load: {state.cognitive_load:.2f}, "
            f"Attention: {state.attention_level:.2f}, "
            f"Fatigue: {state.fatigue_index:.2f}, "
            f"Confidence: {state.confidence:.2f}"
        )

        return state

    async def _estimate_with_xgc_avis(
        self,
        signals: List[UserSignal],
        context: Optional[Dict[str, Any]] = None
    ) -> CognitiveState:
        """Estimate using XGC-AVis service"""
        async with self.xgc_client as client:
            return await client.estimate_cognitive_state(signals, context)

    async def _estimate_heuristic(
        self,
        signals: List[UserSignal],
        context: Optional[Dict[str, Any]] = None
    ) -> CognitiveState:
        """
        Estimate using built-in heuristics

        This is a simplified heuristic model. In production, this would be
        replaced with a more sophisticated ML model.
        """
        # Group signals by type
        signal_dict: Dict[str, List[float]] = {}
        for signal in signals:
            sig_type = signal.signal_type.value
            if sig_type not in signal_dict:
                signal_dict[sig_type] = []
            signal_dict[sig_type].append(signal.normalized_value)

        # Average values per signal type
        avg_signals = {
            sig_type: np.mean(values)
            for sig_type, values in signal_dict.items()
        }

        # Heuristic estimation
        # These are simplified mappings - would be ML model in production

        # Cognitive load: High when interaction timing is slow/erratic
        cognitive_load = avg_signals.get("interaction_timing", 0.5)
        if "mouse_movement" in avg_signals:
            # Erratic mouse movement suggests high cognitive load
            cognitive_load = (cognitive_load + avg_signals["mouse_movement"]) / 2

        # Attention: Based on eye tracking and interaction patterns
        attention_level = 1.0 - avg_signals.get("eye_tracking", 0.5)
        if "interaction_timing" in avg_signals:
            # Consistent timing suggests good attention
            attention_level = (attention_level + (1 - avg_signals["interaction_timing"])) / 2

        # Fatigue: Combination of interaction speed and consistency
        fatigue_index = avg_signals.get("interaction_timing", 0.5)
        if "speech_patterns" in avg_signals:
            # Slow speech suggests fatigue
            fatigue_index = (fatigue_index + avg_signals["speech_patterns"]) / 2

        # Stress: Based on interaction patterns and device orientation changes
        stress_level = avg_signals.get("device_orientation", 0.5)
        if "keyboard_patterns" in avg_signals:
            # Erratic typing suggests stress
            stress_level = (stress_level + avg_signals["keyboard_patterns"]) / 2

        # Reading comprehension: Inverse of cognitive load + attention
        reading_comprehension = (
            (1 - cognitive_load) * 0.4 +
            attention_level * 0.6
        )

        # Calculate confidence based on signal diversity and quantity
        confidence = min(1.0, len(signals) / 10.0) * min(1.0, len(signal_dict) / 3.0)

        return CognitiveState(
            cognitive_load=float(np.clip(cognitive_load, 0.0, 1.0)),
            attention_level=float(np.clip(attention_level, 0.0, 1.0)),
            fatigue_index=float(np.clip(fatigue_index, 0.0, 1.0)),
            stress_level=float(np.clip(stress_level, 0.0, 1.0)),
            reading_comprehension=float(np.clip(reading_comprehension, 0.0, 1.0)),
            confidence=float(confidence)
        )

    def _get_default_state(self) -> CognitiveState:
        """Get default cognitive state"""
        return CognitiveState(
            cognitive_load=0.5,
            attention_level=0.5,
            fatigue_index=0.5,
            stress_level=0.5,
            reading_comprehension=0.5,
            confidence=0.0
        )

    def get_state_trend(
        self,
        dimension: str,
        window_size: int = 10
    ) -> Optional[str]:
        """
        Get trend for a specific state dimension

        Args:
            dimension: State dimension name
            window_size: Number of recent states to consider

        Returns:
            "increasing", "decreasing", "stable", or None
        """
        if len(self.state_history) < window_size:
            return None

        recent_states = list(self.state_history)[-window_size:]
        values = [getattr(state, dimension) for state in recent_states]

        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        threshold = 0.01
        if slope > threshold:
            return "increasing"
        elif slope < -threshold:
            return "decreasing"
        else:
            return "stable"

    def get_average_state(
        self,
        time_window_seconds: Optional[float] = None
    ) -> Optional[CognitiveState]:
        """
        Get average cognitive state over a time window

        Args:
            time_window_seconds: Time window in seconds, or None for all history

        Returns:
            Average CognitiveState or None if no history
        """
        if not self.state_history:
            return None

        if time_window_seconds is None:
            states = list(self.state_history)
        else:
            cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
            states = [
                state for state in self.state_history
                if state.timestamp >= cutoff_time
            ]

        if not states:
            return None

        return CognitiveState(
            cognitive_load=float(np.mean([s.cognitive_load for s in states])),
            attention_level=float(np.mean([s.attention_level for s in states])),
            fatigue_index=float(np.mean([s.fatigue_index for s in states])),
            stress_level=float(np.mean([s.stress_level for s in states])),
            reading_comprehension=float(np.mean([s.reading_comprehension for s in states])),
            confidence=float(np.mean([s.confidence for s in states]))
        )

    async def close(self):
        """Clean up resources"""
        if self.xgc_client.session:
            await self.xgc_client.session.close()
