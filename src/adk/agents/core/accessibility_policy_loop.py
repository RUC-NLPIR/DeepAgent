"""
Accessibility Policy Loop

Coordinates Loop C (Content Refinement), UI Adaptation, and CMS (Memory)
to generate and apply accessibility adaptations.
"""

import asyncio
from typing import Dict, List, Optional, Any

from ...utils.schemas import (
    CognitiveState,
    AccessibilityProfile,
    AccessibilityAdaptation
)
from ...utils.logger import get_logger
from ...tools.memory.memory_manager import MemoryManager
from ..loop_c.refinement_coordinator import RefinementCoordinator
from ..ui_adaptation_agent import UiAdaptationAgent


class AccessibilityPolicyLoop:
    """
    Accessibility Policy Loop

    Generates and applies accessibility policies based on cognitive state
    and user preferences, with content refinement.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AccessibilityPolicyLoop"""
        self.config = config or {}
        self.logger = get_logger("system")

        # Initialize components
        self.memory_manager = MemoryManager(config)
        self.refinement_coordinator = RefinementCoordinator(config)
        self.ui_adaptation_agent = UiAdaptationAgent(config)

        self.logger.info("AccessibilityPolicyLoop initialized")

    async def generate_and_apply_adaptations(
        self,
        cognitive_state: CognitiveState,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        content_to_refine: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate and apply accessibility adaptations

        Args:
            cognitive_state: Current cognitive state
            user_id: User identifier
            session_id: Session identifier
            content_to_refine: Optional content to refine
            context: Optional context

        Returns:
            Dictionary with adaptations and refinement results
        """
        result = {
            "ui_adaptations": [],
            "content_refinement": None,
            "timestamp": datetime.now().isoformat()
        }

        # Retrieve user profile from memory
        accessibility_profile = None
        if user_id:
            accessibility_profile = await self.memory_manager.get_accessibility_profile(user_id)

        # Generate UI adaptations
        ui_adaptations = await self.ui_adaptation_agent.generate_adaptations(
            cognitive_state,
            accessibility_profile,
            context
        )
        result["ui_adaptations"] = ui_adaptations

        # Save adaptation history
        if user_id and session_id:
            for adaptation in ui_adaptations:
                await self.memory_manager.save_adaptation_history(
                    user_id,
                    session_id,
                    adaptation,
                    cognitive_state
                )

        # Refine content if provided
        if content_to_refine:
            refinement_result = await self.refinement_coordinator.refine_content(
                content_to_refine,
                cognitive_state,
                accessibility_profile,
                context
            )
            result["content_refinement"] = refinement_result

        # Save cognitive profile
        if user_id:
            await self.memory_manager.save_cognitive_profile(
                user_id,
                cognitive_state,
                session_id
            )

        return result
