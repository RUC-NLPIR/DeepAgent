"""
Accessibility Coordinator - Top-Level Orchestrator

Coordinates the complete neuroadaptive accessibility system including:
- PerceptionPipeline (Loops A & B)
- AccessibilityPolicyLoop (Loop C, UI Adaptation, CMS)
- LoggingAndEvalAgent (Loop E)
- LoopStopChecker
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from ...utils.schemas import SignalType, EvaluationMetrics
from ...utils.logger import get_logger
from .perception_pipeline import PerceptionPipeline
from .accessibility_policy_loop import AccessibilityPolicyLoop
from ..loop_e.logging_eval_agent import LoggingAndEvalAgent
from ..loop_e.loop_stop_checker import LoopStopChecker


class AccessibilityCoordinator:
    """
    Top-level coordinator for the neuroadaptive accessibility system

    Orchestrates the complete agent hierarchy to provide real-time
    accessibility adaptations based on user cognitive state.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AccessibilityCoordinator"""
        self.config = config or {}
        self.logger = get_logger("system")

        # Initialize major components
        self.perception_pipeline = PerceptionPipeline(config)
        self.accessibility_policy_loop = AccessibilityPolicyLoop(config)
        self.logging_eval_agent = LoggingAndEvalAgent(config)
        self.loop_stop_checker = LoopStopChecker(config)

        # Session tracking
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None

        self.logger.info("AccessibilityCoordinator initialized")

    async def initialize(self):
        """Initialize the coordinator and all sub-components"""
        await self.perception_pipeline.initialize()
        await self.logging_eval_agent.log_system_event(
            "SYSTEM_INIT",
            "AccessibilityCoordinator initialized and ready"
        )
        self.logger.info("AccessibilityCoordinator ready")

    async def start_session(self, user_id: Optional[str] = None) -> str:
        """
        Start a new accessibility session

        Args:
            user_id: Optional user identifier

        Returns:
            Session ID
        """
        self.current_session_id = f"session_{uuid.uuid4().hex[:12]}"
        self.session_start_time = datetime.now()

        await self.logging_eval_agent.log_system_event(
            "SESSION_START",
            f"Started session {self.current_session_id}",
            metadata={"user_id": user_id}
        )

        return self.current_session_id

    async def process_user_interaction(
        self,
        raw_signals: List[tuple[SignalType, Any, Optional[Dict[str, Any]]]],
        user_id: Optional[str] = None,
        content_to_refine: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user interaction through the complete pipeline

        Args:
            raw_signals: Raw user signals
            user_id: User identifier
            content_to_refine: Optional content to refine
            context: Optional context

        Returns:
            Complete processing result
        """
        if not self.current_session_id:
            await self.start_session(user_id)

        interaction_start = datetime.now()

        # Step 1: Perception Pipeline (Loops A & B)
        normalized_signals, cognitive_state = await self.perception_pipeline.process_signals(
            raw_signals,
            context
        )

        # Step 2: Accessibility Policy Loop (Loop C, UI Adaptation, CMS)
        adaptations_result = await self.accessibility_policy_loop.generate_and_apply_adaptations(
            cognitive_state,
            user_id,
            self.current_session_id,
            content_to_refine,
            context
        )

        # Step 3: Calculate metrics and log (Loop E)
        interaction_time = (datetime.now() - interaction_start).total_seconds() * 1000

        metrics = EvaluationMetrics(
            session_id=self.current_session_id,
            adaptation_latency_ms=interaction_time,
            accessibility_score=cognitive_state.confidence,
            refinement_iterations=adaptations_result.get("content_refinement", {}).get("iterations_completed", 0),
            total_adaptations=len(adaptations_result["ui_adaptations"]),
            successful_adaptations=len(adaptations_result["ui_adaptations"])
        )

        await self.logging_eval_agent.log_evaluation_metrics(
            self.current_session_id,
            metrics
        )

        # Compile result
        result = {
            "session_id": self.current_session_id,
            "cognitive_state": {
                "cognitive_load": cognitive_state.cognitive_load,
                "attention_level": cognitive_state.attention_level,
                "fatigue_index": cognitive_state.fatigue_index,
                "stress_level": cognitive_state.stress_level,
                "reading_comprehension": cognitive_state.reading_comprehension,
                "confidence": cognitive_state.confidence
            },
            "ui_adaptations": [
                {
                    "category": a.category,
                    "parameter": a.parameter,
                    "value": a.value,
                    "rationale": a.rationale,
                    "priority": a.priority
                }
                for a in adaptations_result["ui_adaptations"]
            ],
            "content_refinement": adaptations_result.get("content_refinement"),
            "metrics": {
                "latency_ms": interaction_time,
                "accessibility_score": metrics.accessibility_score
            }
        }

        return result

    async def run_adaptive_loop(
        self,
        user_id: str,
        signal_stream: asyncio.Queue,
        max_duration_seconds: Optional[float] = None
    ):
        """
        Run continuous adaptive loop processing signal stream

        Args:
            user_id: User identifier
            signal_stream: Queue of raw signals
            max_duration_seconds: Optional maximum duration
        """
        await self.start_session(user_id)
        iterations = 0
        convergence_scores = []

        while True:
            try:
                # Get signals from stream (with timeout)
                raw_signals = await asyncio.wait_for(
                    signal_stream.get(),
                    timeout=1.0
                )

                # Process interaction
                result = await self.process_user_interaction(
                    raw_signals,
                    user_id
                )

                iterations += 1
                convergence_scores.append(result["cognitive_state"]["confidence"])

                # Check stop conditions
                avg_convergence = sum(convergence_scores[-5:]) / min(5, len(convergence_scores))
                stop_decision = await self.loop_stop_checker.should_stop(
                    iterations,
                    avg_convergence,
                    self.session_start_time
                )

                if stop_decision.should_stop:
                    await self.logging_eval_agent.log_system_event(
                        "LOOP_STOP",
                        stop_decision.reason,
                        metadata={"iterations": iterations}
                    )
                    break

            except asyncio.TimeoutError:
                # No signals received, continue waiting
                continue
            except Exception as e:
                await self.logging_eval_agent.log_system_event(
                    "ERROR",
                    f"Error in adaptive loop: {e}",
                    level="ERROR"
                )
                break

    async def end_session(self) -> Dict[str, Any]:
        """
        End the current session and return statistics

        Returns:
            Session statistics
        """
        if not self.current_session_id:
            return {}

        stats = await self.logging_eval_agent.get_session_statistics(
            self.current_session_id
        )

        await self.logging_eval_agent.log_system_event(
            "SESSION_END",
            f"Ended session {self.current_session_id}",
            metadata=stats
        )

        session_id = self.current_session_id
        self.current_session_id = None
        self.session_start_time = None

        return {
            "session_id": session_id,
            "statistics": stats
        }

    async def close(self):
        """Clean up all resources"""
        await self.perception_pipeline.close()
        await self.logging_eval_agent.log_system_event(
            "SYSTEM_SHUTDOWN",
            "AccessibilityCoordinator shutting down"
        )
