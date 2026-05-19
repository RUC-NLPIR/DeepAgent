"""
UI Adaptation Agent

Generates real-time UI adaptation recommendations based on cognitive state
and accessibility profiles.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..utils.schemas import (
    CognitiveState,
    AccessibilityProfile,
    AccessibilityAdaptation
)
from ..utils.config_loader import get_config_value
from ..utils.logger import get_logger


class UiAdaptationAgent:
    """
    Agent for generating UI adaptations

    Monitors cognitive state and user preferences to generate real-time
    UI adaptation recommendations for accessibility.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the UiAdaptationAgent"""
        self.config = config or {}
        self.logger = get_logger("system")

        self.enabled = get_config_value("ui_adaptation.enabled", True)
        self.adaptation_categories = get_config_value(
            "ui_adaptation.adaptation_categories",
            ["text_size", "contrast", "color_scheme", "layout_density"]
        )
        self.real_time_updates = get_config_value("ui_adaptation.real_time_updates", True)
        self.debounce_ms = get_config_value("ui_adaptation.debounce_ms", 200)

        self.last_adaptation_time: Optional[datetime] = None
        self.current_adaptations: List[AccessibilityAdaptation] = []

        self.logger.info(f"UiAdaptationAgent initialized (categories: {len(self.adaptation_categories)})")

    async def generate_adaptations(
        self,
        cognitive_state: CognitiveState,
        accessibility_profile: Optional[AccessibilityProfile] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[AccessibilityAdaptation]:
        """
        Generate UI adaptations based on current state

        Args:
            cognitive_state: Current cognitive state
            accessibility_profile: User's accessibility profile
            context: Optional context (current page, task, etc.)

        Returns:
            List of AccessibilityAdaptation recommendations
        """
        if not self.enabled:
            return []

        # Debounce: Check if enough time has passed
        if self.last_adaptation_time and self.real_time_updates:
            elapsed_ms = (datetime.now() - self.last_adaptation_time).total_seconds() * 1000
            if elapsed_ms < self.debounce_ms:
                return self.current_adaptations

        self.logger.debug("Generating UI adaptations")

        adaptations = []

        # Generate adaptations for each category
        for category in self.adaptation_categories:
            category_adaptations = await self._generate_category_adaptations(
                category,
                cognitive_state,
                accessibility_profile,
                context
            )
            adaptations.extend(category_adaptations)

        # Sort by priority
        adaptations.sort(key=lambda x: x.priority, reverse=True)

        # Update tracking
        self.current_adaptations = adaptations
        self.last_adaptation_time = datetime.now()

        self.logger.debug(f"Generated {len(adaptations)} UI adaptations")

        return adaptations

    async def _generate_category_adaptations(
        self,
        category: str,
        cognitive_state: CognitiveState,
        accessibility_profile: Optional[AccessibilityProfile],
        context: Optional[Dict[str, Any]]
    ) -> List[AccessibilityAdaptation]:
        """Generate adaptations for a specific category"""
        adaptations = []

        if category == "text_size":
            adaptations.extend(self._adapt_text_size(cognitive_state, accessibility_profile))
        elif category == "contrast":
            adaptations.extend(self._adapt_contrast(cognitive_state, accessibility_profile))
        elif category == "color_scheme":
            adaptations.extend(self._adapt_color_scheme(cognitive_state, accessibility_profile))
        elif category == "layout_density":
            adaptations.extend(self._adapt_layout_density(cognitive_state, accessibility_profile))
        elif category == "animation_speed":
            adaptations.extend(self._adapt_animation_speed(cognitive_state, accessibility_profile))
        elif category == "audio_descriptions":
            adaptations.extend(self._adapt_audio(cognitive_state, accessibility_profile))
        elif category == "simplified_language":
            adaptations.extend(self._adapt_language(cognitive_state, accessibility_profile))

        return adaptations

    def _adapt_text_size(
        self,
        cognitive_state: CognitiveState,
        profile: Optional[AccessibilityProfile]
    ) -> List[AccessibilityAdaptation]:
        """Generate text size adaptations"""
        adaptations = []

        # Base size from profile
        base_size = 1.0
        if profile and "text_size" in profile.settings:
            base_size = profile.settings["text_size"]

        # Adjust based on cognitive state
        size_multiplier = base_size

        # High cognitive load -> larger text
        if cognitive_state.cognitive_load > 0.7:
            size_multiplier *= 1.15
            adaptations.append(AccessibilityAdaptation(
                adaptation_id=f"text_size_{datetime.now().timestamp()}",
                category="text_size",
                parameter="font_size_multiplier",
                value=size_multiplier,
                confidence=cognitive_state.confidence * 0.9,
                rationale="Increased text size due to high cognitive load",
                priority=8
            ))

        # High fatigue -> larger text
        elif cognitive_state.fatigue_index > 0.7:
            size_multiplier *= 1.1
            adaptations.append(AccessibilityAdaptation(
                adaptation_id=f"text_size_{datetime.now().timestamp()}",
                category="text_size",
                parameter="font_size_multiplier",
                value=size_multiplier,
                confidence=cognitive_state.confidence * 0.85,
                rationale="Increased text size due to fatigue",
                priority=7
            ))

        return adaptations

    def _adapt_contrast(
        self,
        cognitive_state: CognitiveState,
        profile: Optional[AccessibilityProfile]
    ) -> List[AccessibilityAdaptation]:
        """Generate contrast adaptations"""
        adaptations = []

        # Check profile preference
        if profile and profile.settings.get("contrast") == "high":
            return adaptations  # Already at high contrast

        # High fatigue or low attention -> increase contrast
        if cognitive_state.fatigue_index > 0.6 or cognitive_state.attention_level < 0.4:
            adaptations.append(AccessibilityAdaptation(
                adaptation_id=f"contrast_{datetime.now().timestamp()}",
                category="contrast",
                parameter="contrast_level",
                value="high",
                confidence=cognitive_state.confidence * 0.8,
                rationale="Increased contrast for better visibility",
                priority=7
            ))

        return adaptations

    def _adapt_color_scheme(
        self,
        cognitive_state: CognitiveState,
        profile: Optional[AccessibilityProfile]
    ) -> List[AccessibilityAdaptation]:
        """Generate color scheme adaptations"""
        adaptations = []

        # Check profile preference
        if profile and "color_scheme" in profile.settings:
            return adaptations  # Use profile preference

        # High stress -> calming colors
        if cognitive_state.stress_level > 0.7:
            adaptations.append(AccessibilityAdaptation(
                adaptation_id=f"color_{datetime.now().timestamp()}",
                category="color_scheme",
                parameter="theme",
                value="calm_blue",
                confidence=cognitive_state.confidence * 0.7,
                rationale="Applied calming color scheme due to high stress",
                priority=5
            ))

        return adaptations

    def _adapt_layout_density(
        self,
        cognitive_state: CognitiveState,
        profile: Optional[AccessibilityProfile]
    ) -> List[AccessibilityAdaptation]:
        """Generate layout density adaptations"""
        adaptations = []

        # High cognitive load -> reduce density
        if cognitive_state.cognitive_load > 0.7:
            adaptations.append(AccessibilityAdaptation(
                adaptation_id=f"layout_{datetime.now().timestamp()}",
                category="layout_density",
                parameter="density",
                value="sparse",
                confidence=cognitive_state.confidence * 0.85,
                rationale="Reduced layout density to decrease cognitive load",
                priority=9
            ))

        return adaptations

    def _adapt_animation_speed(
        self,
        cognitive_state: CognitiveState,
        profile: Optional[AccessibilityProfile]
    ) -> List[AccessibilityAdaptation]:
        """Generate animation speed adaptations"""
        adaptations = []

        # High cognitive load or fatigue -> slow animations
        if cognitive_state.cognitive_load > 0.7 or cognitive_state.fatigue_index > 0.7:
            adaptations.append(AccessibilityAdaptation(
                adaptation_id=f"animation_{datetime.now().timestamp()}",
                category="animation_speed",
                parameter="animation_duration_multiplier",
                value=1.5,  # 50% slower
                confidence=cognitive_state.confidence * 0.75,
                rationale="Slowed animations to reduce cognitive demand",
                priority=4
            ))

        return adaptations

    def _adapt_audio(
        self,
        cognitive_state: CognitiveState,
        profile: Optional[AccessibilityProfile]
    ) -> List[AccessibilityAdaptation]:
        """Generate audio description adaptations"""
        adaptations = []

        # Low reading comprehension -> enable audio
        if cognitive_state.reading_comprehension < 0.4:
            adaptations.append(AccessibilityAdaptation(
                adaptation_id=f"audio_{datetime.now().timestamp()}",
                category="audio_descriptions",
                parameter="enable_audio_descriptions",
                value=True,
                confidence=cognitive_state.confidence * 0.9,
                rationale="Enabled audio descriptions due to low reading comprehension",
                priority=10
            ))

        return adaptations

    def _adapt_language(
        self,
        cognitive_state: CognitiveState,
        profile: Optional[AccessibilityProfile]
    ) -> List[AccessibilityAdaptation]:
        """Generate language simplification adaptations"""
        adaptations = []

        # Low reading comprehension -> simplify language
        if cognitive_state.reading_comprehension < 0.5:
            adaptations.append(AccessibilityAdaptation(
                adaptation_id=f"language_{datetime.now().timestamp()}",
                category="simplified_language",
                parameter="enable_simplified_language",
                value=True,
                confidence=cognitive_state.confidence * 0.95,
                rationale="Enabled simplified language due to low reading comprehension",
                priority=10
            ))

        return adaptations

    async def apply_adaptations(
        self,
        adaptations: List[AccessibilityAdaptation],
        ui_client: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Apply adaptations to UI (interface for UI client)

        Args:
            adaptations: List of adaptations to apply
            ui_client: Optional UI client for applying changes

        Returns:
            Dictionary with application results
        """
        results = {
            "total_adaptations": len(adaptations),
            "applied": [],
            "failed": [],
            "timestamp": datetime.now().isoformat()
        }

        for adaptation in adaptations:
            try:
                if ui_client:
                    # In production, would call UI client methods
                    # await ui_client.apply_adaptation(adaptation)
                    pass

                results["applied"].append({
                    "adaptation_id": adaptation.adaptation_id,
                    "category": adaptation.category,
                    "parameter": adaptation.parameter,
                    "value": adaptation.value
                })

                self.logger.debug(f"Applied adaptation: {adaptation.category}/{adaptation.parameter}")

            except Exception as e:
                self.logger.error(f"Failed to apply adaptation {adaptation.adaptation_id}: {e}")
                results["failed"].append({
                    "adaptation_id": adaptation.adaptation_id,
                    "error": str(e)
                })

        return results
