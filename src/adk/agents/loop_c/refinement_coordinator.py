"""
Refinement Coordinator - Loop C Meta-Agent

Coordinates the three specialist agents (Factuality, Personalization, Coherence)
to iteratively refine content until convergence or timeout.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...utils.schemas import ContentRefinement, CognitiveState, AccessibilityProfile
from ...utils.config_loader import get_config_value
from ...utils.logger import get_logger
from .factuality_agent import FactualityAgent
from .personalization_agent import PersonalizationAgent
from .coherence_agent import CoherenceAgent


class RefinementCoordinator:
    """
    Meta-agent that coordinates content refinement specialists

    Orchestrates iterative refinement through Factuality, Personalization,
    and Coherence agents until quality convergence or timeout.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the RefinementCoordinator"""
        self.config = config or {}
        self.logger = get_logger("system")

        # Load configuration
        self.max_iterations = get_config_value("loop_c.refinement_coordinator.max_iterations", 5)
        self.convergence_threshold = get_config_value(
            "loop_c.refinement_coordinator.convergence_threshold",
            0.95
        )
        self.timeout_seconds = get_config_value(
            "loop_c.refinement_coordinator.timeout_seconds",
            30
        )

        # Initialize specialist agents
        self.factuality_agent = FactualityAgent(config)
        self.personalization_agent = PersonalizationAgent(config)
        self.coherence_agent = CoherenceAgent(config)

        self.logger.info(f"RefinementCoordinator initialized (max_iter: {self.max_iterations})")

    async def refine_content(
        self,
        content: str,
        cognitive_state: Optional[CognitiveState] = None,
        accessibility_profile: Optional[AccessibilityProfile] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate content refinement through specialists

        Args:
            content: Original content
            cognitive_state: User's cognitive state
            accessibility_profile: User's accessibility profile
            context: Optional context

        Returns:
            Dictionary with final refined content and metadata
        """
        self.logger.debug("Starting coordinated content refinement")
        start_time = datetime.now()

        current_content = content
        refinement_history = []
        all_changes = []

        # Iterative refinement loop
        for iteration in range(self.max_iterations):
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.timeout_seconds:
                self.logger.warning(f"Refinement timeout after {iteration} iterations")
                break

            iteration_start = datetime.now()
            iteration_changes = []

            # Step 1: Factuality refinement
            factuality_result = await self.factuality_agent.refine_content(
                current_content,
                context
            )
            current_content = factuality_result.refined_content
            iteration_changes.extend([f"[Factuality] {c}" for c in factuality_result.changes_made])

            # Step 2: Personalization refinement
            personalization_result = await self.personalization_agent.refine_content(
                current_content,
                cognitive_state,
                accessibility_profile,
                context
            )
            current_content = personalization_result.refined_content
            iteration_changes.extend([f"[Personalization] {c}" for c in personalization_result.changes_made])

            # Step 3: Coherence refinement
            coherence_result = await self.coherence_agent.refine_content(
                current_content,
                context
            )
            current_content = coherence_result.refined_content
            iteration_changes.extend([f"[Coherence] {c}" for c in coherence_result.changes_made])

            # Calculate combined quality score
            combined_score = (
                factuality_result.quality_score * 0.35 +
                personalization_result.quality_score * 0.35 +
                coherence_result.quality_score * 0.30
            )

            iteration_time = (datetime.now() - iteration_start).total_seconds()

            # Record iteration
            iteration_record = {
                "iteration": iteration + 1,
                "content": current_content,
                "factuality_score": factuality_result.quality_score,
                "personalization_score": personalization_result.quality_score,
                "coherence_score": coherence_result.quality_score,
                "combined_score": combined_score,
                "changes": iteration_changes,
                "duration_seconds": iteration_time
            }
            refinement_history.append(iteration_record)
            all_changes.extend(iteration_changes)

            self.logger.debug(
                f"Iteration {iteration + 1}: Score={combined_score:.3f}, "
                f"Changes={len(iteration_changes)}"
            )

            # Check convergence
            if combined_score >= self.convergence_threshold:
                self.logger.info(f"Convergence achieved at iteration {iteration + 1}")
                break

            # Check if no changes were made (local minimum)
            if not iteration_changes:
                self.logger.info(f"No further changes at iteration {iteration + 1}")
                break

        # Final result
        total_time = (datetime.now() - start_time).total_seconds()
        final_iteration = refinement_history[-1] if refinement_history else None

        result = {
            "original_content": content,
            "refined_content": current_content,
            "iterations_completed": len(refinement_history),
            "final_quality_score": final_iteration["combined_score"] if final_iteration else 0.0,
            "total_changes": len(all_changes),
            "total_duration_seconds": total_time,
            "converged": final_iteration["combined_score"] >= self.convergence_threshold if final_iteration else False,
            "refinement_history": refinement_history,
            "all_changes": all_changes,
            "metadata": {
                "max_iterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold,
                "timeout_seconds": self.timeout_seconds
            }
        }

        self.logger.info(
            f"Refinement complete: {len(refinement_history)} iterations, "
            f"Final score: {result['final_quality_score']:.3f}, "
            f"Time: {total_time:.2f}s"
        )

        return result

    async def batch_refine(
        self,
        contents: List[str],
        cognitive_states: Optional[List[CognitiveState]] = None,
        accessibility_profiles: Optional[List[AccessibilityProfile]] = None,
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Refine multiple contents in parallel"""
        # Prepare arguments
        n = len(contents)
        if cognitive_states is None:
            cognitive_states = [None] * n
        if accessibility_profiles is None:
            accessibility_profiles = [None] * n
        if contexts is None:
            contexts = [None] * n

        # Create tasks
        tasks = [
            self.refine_content(content, cog_state, profile, ctx)
            for content, cog_state, profile, ctx in zip(
                contents, cognitive_states, accessibility_profiles, contexts
            )
        ]

        return await asyncio.gather(*tasks)
