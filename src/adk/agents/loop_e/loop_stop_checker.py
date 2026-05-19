"""
Loop Stop Checker

Determines when to stop the agent processing loop based on convergence,
timeout, user satisfaction, or other stopping conditions.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ...utils.schemas import LoopStopDecision, EvaluationMetrics
from ...utils.config_loader import get_config_value
from ...utils.logger import get_logger


class LoopStopChecker:
    """
    Agent for determining when to stop the processing loop

    Checks multiple stop conditions:
    - Maximum iterations reached
    - Quality convergence achieved
    - Timeout exceeded
    - User satisfaction threshold met
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LoopStopChecker"""
        self.config = config or {}
        self.logger = get_logger("system")

        self.enabled = get_config_value("loop_stop.enabled", True)
        self.max_iterations = get_config_value("loop_stop.stop_conditions.max_iterations", 10)
        self.convergence_threshold = get_config_value(
            "loop_stop.stop_conditions.convergence_threshold",
            0.98
        )
        self.timeout_seconds = get_config_value(
            "loop_stop.stop_conditions.timeout_seconds",
            60
        )
        self.user_satisfaction_threshold = get_config_value(
            "loop_stop.stop_conditions.user_satisfaction_threshold",
            0.9
        )
        self.graceful_degradation = get_config_value(
            "loop_stop.graceful_degradation",
            True
        )

        self.logger.info(f"LoopStopChecker initialized (max_iter: {self.max_iterations})")

    async def should_stop(
        self,
        iterations_completed: int,
        convergence_score: float,
        start_time: datetime,
        user_satisfaction: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LoopStopDecision:
        """
        Check if processing loop should stop

        Args:
            iterations_completed: Number of iterations completed
            convergence_score: Current convergence/quality score
            start_time: Loop start time
            user_satisfaction: Optional user satisfaction score
            metadata: Optional additional metadata

        Returns:
            LoopStopDecision with stop decision and reason
        """
        if not self.enabled:
            return LoopStopDecision(
                should_stop=False,
                reason="LoopStopChecker disabled",
                iterations_completed=iterations_completed,
                convergence_score=convergence_score,
                elapsed_time_seconds=0.0
            )

        elapsed_time = (datetime.now() - start_time).total_seconds()

        # Check max iterations
        if iterations_completed >= self.max_iterations:
            return LoopStopDecision(
                should_stop=True,
                reason=f"Maximum iterations reached ({self.max_iterations})",
                iterations_completed=iterations_completed,
                convergence_score=convergence_score,
                elapsed_time_seconds=elapsed_time,
                metadata=metadata or {}
            )

        # Check convergence
        if convergence_score >= self.convergence_threshold:
            return LoopStopDecision(
                should_stop=True,
                reason=f"Convergence achieved (score: {convergence_score:.3f})",
                iterations_completed=iterations_completed,
                convergence_score=convergence_score,
                elapsed_time_seconds=elapsed_time,
                metadata=metadata or {}
            )

        # Check timeout
        if elapsed_time >= self.timeout_seconds:
            if self.graceful_degradation:
                return LoopStopDecision(
                    should_stop=True,
                    reason=f"Timeout reached ({elapsed_time:.1f}s), graceful degradation",
                    iterations_completed=iterations_completed,
                    convergence_score=convergence_score,
                    elapsed_time_seconds=elapsed_time,
                    metadata=metadata or {}
                )

        # Check user satisfaction
        if user_satisfaction is not None and user_satisfaction >= self.user_satisfaction_threshold:
            return LoopStopDecision(
                should_stop=True,
                reason=f"User satisfaction threshold met ({user_satisfaction:.3f})",
                iterations_completed=iterations_completed,
                convergence_score=convergence_score,
                elapsed_time_seconds=elapsed_time,
                metadata=metadata or {}
            )

        # Continue processing
        return LoopStopDecision(
            should_stop=False,
            reason="No stop condition met, continuing",
            iterations_completed=iterations_completed,
            convergence_score=convergence_score,
            elapsed_time_seconds=elapsed_time,
            metadata=metadata or {}
        )

    async def estimate_remaining_time(
        self,
        iterations_completed: int,
        elapsed_time: float
    ) -> Optional[float]:
        """
        Estimate remaining time to completion

        Args:
            iterations_completed: Iterations completed so far
            elapsed_time: Time elapsed so far

        Returns:
            Estimated remaining seconds, or None if cannot estimate
        """
        if iterations_completed == 0:
            return None

        avg_time_per_iteration = elapsed_time / iterations_completed
        remaining_iterations = max(0, self.max_iterations - iterations_completed)

        return avg_time_per_iteration * remaining_iterations
