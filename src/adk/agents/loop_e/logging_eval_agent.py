"""
Logging and Evaluation Agent - Loop E

Dual-logging system for system events and evaluation metrics.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...utils.schemas import EvaluationMetrics, AgentState
from ...utils.config_loader import get_config_value
from ...utils.logger import get_logger, setup_logging


class LoggingAndEvalAgent:
    """
    Agent for logging and evaluation

    Maintains dual logging:
    1. System log: Operational events, errors, debugging
    2. Evaluation log: Performance metrics, quality scores, adaptations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LoggingAndEvalAgent"""
        self.config = config or {}

        # Setup dual logging
        self.system_logger, self.eval_logger = setup_logging(
            log_dir=get_config_value("loop_e.dual_logging.system_log.path", "logs").split('/')[0],
            system_log_level=get_config_value("loop_e.dual_logging.system_log.level", "INFO"),
            eval_log_level=get_config_value("loop_e.dual_logging.evaluation_log.level", "DEBUG")
        )

        self.enabled = get_config_value("loop_e.enabled", True)
        self.metrics_enabled = get_config_value("loop_e.dual_logging.evaluation_log.include_metrics", True)

        # Metrics tracking
        self.session_metrics: Dict[str, List[EvaluationMetrics]] = {}
        self.agent_states: Dict[str, AgentState] = {}

        self.system_logger.info("LoggingAndEvalAgent initialized")

    async def log_system_event(
        self,
        event_type: str,
        message: str,
        level: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a system event"""
        log_message = f"[{event_type}] {message}"
        if metadata:
            log_message += f" | Metadata: {metadata}"

        if level == "DEBUG":
            self.system_logger.debug(log_message)
        elif level == "INFO":
            self.system_logger.info(log_message)
        elif level == "WARNING":
            self.system_logger.warning(log_message)
        elif level == "ERROR":
            self.system_logger.error(log_message)

    async def log_evaluation_metrics(
        self,
        session_id: str,
        metrics: EvaluationMetrics
    ):
        """Log evaluation metrics"""
        if not self.metrics_enabled:
            return

        # Store metrics
        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = []
        self.session_metrics[session_id].append(metrics)

        # Log to evaluation logger
        self.eval_logger.info(
            f"Session: {session_id} | "
            f"Latency: {metrics.adaptation_latency_ms:.2f}ms | "
            f"Accessibility Score: {metrics.accessibility_score:.3f} | "
            f"Iterations: {metrics.refinement_iterations} | "
            f"Adaptations: {metrics.successful_adaptations}/{metrics.total_adaptations}"
        )

    async def update_agent_state(
        self,
        agent_id: str,
        agent_type: str,
        status: str,
        current_task: Optional[str] = None,
        progress: float = 0.0,
        error_message: Optional[str] = None
    ):
        """Update agent state"""
        state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            status=status,
            current_task=current_task,
            progress=progress,
            error_message=error_message
        )

        self.agent_states[agent_id] = state

        self.system_logger.debug(
            f"Agent state updated: {agent_id} ({agent_type}) -> {status}"
        )

    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        if session_id not in self.session_metrics:
            return {}

        metrics_list = self.session_metrics[session_id]
        if not metrics_list:
            return {}

        import numpy as np

        return {
            "session_id": session_id,
            "total_metrics": len(metrics_list),
            "avg_latency_ms": float(np.mean([m.adaptation_latency_ms for m in metrics_list])),
            "avg_accessibility_score": float(np.mean([m.accessibility_score for m in metrics_list])),
            "total_adaptations": sum(m.total_adaptations for m in metrics_list),
            "successful_adaptations": sum(m.successful_adaptations for m in metrics_list),
            "success_rate": sum(m.successful_adaptations for m in metrics_list) / max(1, sum(m.total_adaptations for m in metrics_list))
        }

    async def get_all_agent_states(self) -> Dict[str, AgentState]:
        """Get states of all agents"""
        return self.agent_states.copy()
