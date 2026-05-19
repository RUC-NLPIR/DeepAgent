"""
SignalNormalizer Agent - Loop A

This agent normalizes raw user signals from various sources (eye tracking,
speech patterns, interaction timing, etc.) into standardized formats suitable
for downstream processing.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque

from ...utils.schemas import UserSignal, SignalType
from ...utils.config_loader import get_config_value
from ...utils.logger import get_logger


class SignalNormalizer:
    """
    Agent for normalizing heterogeneous user signals

    This agent processes raw signals from various sources and normalizes them
    to a common scale (0-1) using configurable normalization strategies.

    Normalization strategies:
    - z_score: Standardization using mean and std
    - min_max: Min-max scaling to [0, 1]
    - robust: Robust scaling using median and IQR
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SignalNormalizer agent

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger("system")

        # Load configuration
        self.enabled = get_config_value("loop_a.enabled", True)
        self.signal_types = get_config_value(
            "loop_a.signal_types",
            ["eye_tracking", "speech_patterns", "interaction_timing"]
        )
        self.normalization_strategy = get_config_value(
            "loop_a.normalization_strategy",
            "z_score"
        )
        self.outlier_threshold = get_config_value(
            "loop_a.outlier_threshold",
            3.0
        )

        # Statistics tracking for each signal type
        self.signal_stats: Dict[str, Dict[str, deque]] = {}
        self.window_size = 100  # Rolling window for statistics

        # Initialize statistics for each signal type
        for signal_type in self.signal_types:
            self.signal_stats[signal_type] = {
                "values": deque(maxlen=self.window_size),
                "timestamps": deque(maxlen=self.window_size)
            }

        self.logger.info(
            f"SignalNormalizer initialized with strategy: {self.normalization_strategy}"
        )

    async def normalize_signal(
        self,
        signal_type: SignalType,
        raw_value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UserSignal:
        """
        Normalize a single signal

        Args:
            signal_type: Type of signal
            raw_value: Raw signal value
            metadata: Optional metadata

        Returns:
            Normalized UserSignal object
        """
        if not self.enabled:
            self.logger.warning("SignalNormalizer is disabled")
            return UserSignal(
                signal_type=signal_type,
                raw_value=raw_value,
                normalized_value=0.0,
                metadata=metadata or {}
            )

        signal_key = signal_type.value

        # Convert raw value to float if needed
        numeric_value = self._extract_numeric_value(raw_value)

        # Update statistics
        self.signal_stats[signal_key]["values"].append(numeric_value)
        self.signal_stats[signal_key]["timestamps"].append(datetime.now())

        # Normalize based on strategy
        if self.normalization_strategy == "z_score":
            normalized = self._z_score_normalize(signal_key, numeric_value)
        elif self.normalization_strategy == "min_max":
            normalized = self._min_max_normalize(signal_key, numeric_value)
        elif self.normalization_strategy == "robust":
            normalized = self._robust_normalize(signal_key, numeric_value)
        else:
            self.logger.warning(
                f"Unknown normalization strategy: {self.normalization_strategy}. "
                "Using z_score."
            )
            normalized = self._z_score_normalize(signal_key, numeric_value)

        # Check for outliers
        is_outlier = self._is_outlier(signal_key, numeric_value)
        if is_outlier:
            self.logger.debug(
                f"Outlier detected for {signal_type}: {numeric_value}"
            )
            if metadata is None:
                metadata = {}
            metadata["outlier"] = True

        # Create UserSignal object
        user_signal = UserSignal(
            signal_type=signal_type,
            raw_value=raw_value,
            normalized_value=float(np.clip(normalized, 0.0, 1.0)),
            metadata=metadata or {}
        )

        self.logger.debug(
            f"Normalized {signal_type}: {raw_value} -> {user_signal.normalized_value}"
        )

        return user_signal

    async def normalize_batch(
        self,
        signals: List[tuple[SignalType, Any, Optional[Dict[str, Any]]]]
    ) -> List[UserSignal]:
        """
        Normalize a batch of signals concurrently

        Args:
            signals: List of (signal_type, raw_value, metadata) tuples

        Returns:
            List of normalized UserSignal objects
        """
        tasks = [
            self.normalize_signal(sig_type, raw_val, meta)
            for sig_type, raw_val, meta in signals
        ]
        return await asyncio.gather(*tasks)

    def _extract_numeric_value(self, raw_value: Any) -> float:
        """Extract numeric value from various input types"""
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        elif isinstance(raw_value, dict):
            # For structured data, use 'value' key or first numeric value
            if "value" in raw_value:
                return float(raw_value["value"])
            for v in raw_value.values():
                if isinstance(v, (int, float)):
                    return float(v)
            return 0.0
        elif isinstance(raw_value, list):
            # Use mean of list
            return float(np.mean([x for x in raw_value if isinstance(x, (int, float))]))
        else:
            try:
                return float(raw_value)
            except (ValueError, TypeError):
                self.logger.warning(f"Could not convert to float: {raw_value}")
                return 0.0

    def _z_score_normalize(self, signal_key: str, value: float) -> float:
        """Normalize using z-score (standardization)"""
        values = list(self.signal_stats[signal_key]["values"])

        if len(values) < 2:
            return 0.5  # Default to middle if not enough data

        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return 0.5

        # Z-score
        z = (value - mean) / std

        # Map to [0, 1] using sigmoid-like function
        normalized = 1 / (1 + np.exp(-z))

        return normalized

    def _min_max_normalize(self, signal_key: str, value: float) -> float:
        """Normalize using min-max scaling"""
        values = list(self.signal_stats[signal_key]["values"])

        if len(values) < 2:
            return 0.5

        min_val = np.min(values)
        max_val = np.max(values)

        if max_val == min_val:
            return 0.5

        normalized = (value - min_val) / (max_val - min_val)

        return normalized

    def _robust_normalize(self, signal_key: str, value: float) -> float:
        """Normalize using robust scaling (median and IQR)"""
        values = list(self.signal_stats[signal_key]["values"])

        if len(values) < 4:
            return 0.5

        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        if iqr == 0:
            return 0.5

        # Robust z-score
        robust_z = (value - median) / iqr

        # Map to [0, 1]
        normalized = 1 / (1 + np.exp(-robust_z))

        return normalized

    def _is_outlier(self, signal_key: str, value: float) -> bool:
        """Check if value is an outlier"""
        values = list(self.signal_stats[signal_key]["values"])

        if len(values) < 10:
            return False

        if self.normalization_strategy == "robust":
            median = np.median(values)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            if iqr == 0:
                return False

            # Modified z-score
            modified_z = abs((value - median) / iqr)
            return modified_z > self.outlier_threshold
        else:
            # Standard z-score
            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                return False

            z = abs((value - mean) / std)
            return z > self.outlier_threshold

    def get_statistics(self, signal_type: Optional[SignalType] = None) -> Dict[str, Any]:
        """
        Get current statistics for signal types

        Args:
            signal_type: Specific signal type, or None for all

        Returns:
            Dictionary of statistics
        """
        if signal_type:
            signal_key = signal_type.value
            values = list(self.signal_stats[signal_key]["values"])

            if not values:
                return {}

            return {
                "signal_type": signal_type.value,
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
        else:
            return {
                sig_key: self.get_statistics(SignalType(sig_key))
                for sig_key in self.signal_stats.keys()
            }

    def reset_statistics(self, signal_type: Optional[SignalType] = None):
        """
        Reset statistics for signal types

        Args:
            signal_type: Specific signal type, or None for all
        """
        if signal_type:
            signal_key = signal_type.value
            self.signal_stats[signal_key]["values"].clear()
            self.signal_stats[signal_key]["timestamps"].clear()
            self.logger.info(f"Reset statistics for {signal_type}")
        else:
            for sig_key in self.signal_stats.keys():
                self.signal_stats[sig_key]["values"].clear()
                self.signal_stats[sig_key]["timestamps"].clear()
            self.logger.info("Reset all signal statistics")
