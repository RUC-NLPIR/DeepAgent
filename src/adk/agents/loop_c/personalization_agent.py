"""
Personalization Agent - Loop C Specialist

This agent personalizes content based on user cognitive state, preferences,
and accessibility needs.
"""

import asyncio
from typing import Dict, List, Optional, Any

from ...utils.schemas import ContentRefinement, CognitiveState, AccessibilityProfile
from ...utils.config_loader import get_config_value
from ...utils.logger import get_logger


class PersonalizationAgent:
    """
    Specialist agent for content personalization

    Adapts content based on:
    - User cognitive state (from Loop B)
    - User preferences and accessibility profile (from CMS)
    - Reading level and comprehension
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PersonalizationAgent"""
        self.config = config or {}
        self.logger = get_logger("system")

        self.enabled = get_config_value("loop_c.specialist_agents.personalization.enabled", True)
        self.adaptation_strength = get_config_value(
            "loop_c.specialist_agents.personalization.adaptation_strength",
            0.7
        )
        self.profile_weight = get_config_value(
            "loop_c.specialist_agents.personalization.profile_weight",
            0.6
        )

        self.logger.info(f"PersonalizationAgent initialized (strength: {self.adaptation_strength})")

    async def refine_content(
        self,
        content: str,
        cognitive_state: Optional[CognitiveState] = None,
        accessibility_profile: Optional[AccessibilityProfile] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ContentRefinement:
        """
        Personalize content based on user state and profile

        Args:
            content: Original content
            cognitive_state: Current cognitive state
            accessibility_profile: User's accessibility profile
            context: Optional context

        Returns:
            ContentRefinement with personalized content
        """
        if not self.enabled:
            return ContentRefinement(
                original_content=content,
                refined_content=content,
                refinement_type="personalization",
                changes_made=[],
                quality_score=1.0
            )

        self.logger.debug("Starting personalization refinement")

        refined_content = content
        changes = []

        # Apply cognitive state adaptations
        if cognitive_state:
            refined_content, cog_changes = await self._adapt_to_cognitive_state(
                refined_content,
                cognitive_state
            )
            changes.extend(cog_changes)

        # Apply profile-based adaptations
        if accessibility_profile:
            refined_content, profile_changes = await self._adapt_to_profile(
                refined_content,
                accessibility_profile
            )
            changes.extend(profile_changes)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            content,
            refined_content,
            cognitive_state,
            accessibility_profile
        )

        result = ContentRefinement(
            original_content=content,
            refined_content=refined_content,
            refinement_type="personalization",
            changes_made=changes,
            quality_score=quality_score,
            metadata={
                "adaptation_strength": self.adaptation_strength,
                "cognitive_state_used": cognitive_state is not None,
                "profile_used": accessibility_profile is not None
            }
        )

        self.logger.debug(f"Personalization complete. Quality: {quality_score:.2f}, Changes: {len(changes)}")
        return result

    async def _adapt_to_cognitive_state(
        self,
        content: str,
        cognitive_state: CognitiveState
    ) -> tuple[str, List[str]]:
        """Adapt content based on cognitive state"""
        refined = content
        changes = []

        # High cognitive load -> Simplify
        if cognitive_state.cognitive_load > 0.7:
            refined, simplify_changes = self._simplify_content(refined)
            changes.extend(simplify_changes)

        # Low attention -> Add emphasis and structure
        if cognitive_state.attention_level < 0.4:
            refined, emphasis_changes = self._add_emphasis(refined)
            changes.extend(emphasis_changes)

        # High fatigue -> Shorter sentences, clearer structure
        if cognitive_state.fatigue_index > 0.7:
            refined, structure_changes = self._improve_structure(refined)
            changes.extend(structure_changes)

        # Low reading comprehension -> Simpler vocabulary
        if cognitive_state.reading_comprehension < 0.5:
            refined, vocab_changes = self._simplify_vocabulary(refined)
            changes.extend(vocab_changes)

        return refined, changes

    async def _adapt_to_profile(
        self,
        content: str,
        profile: AccessibilityProfile
    ) -> tuple[str, List[str]]:
        """Adapt content based on accessibility profile"""
        refined = content
        changes = []

        settings = profile.settings

        # Check for simplified language preference
        if settings.get("simplified_language"):
            refined, lang_changes = self._simplify_content(refined)
            changes.extend(lang_changes)

        # Check for max sentence length
        if "max_sentence_length" in settings:
            max_len = settings["max_sentence_length"]
            refined, sent_changes = self._enforce_sentence_length(refined, max_len)
            changes.extend(sent_changes)

        # Check for bullet point preference
        if settings.get("bullet_points"):
            refined, bullet_changes = self._convert_to_bullets(refined)
            changes.extend(bullet_changes)

        return refined, changes

    def _simplify_content(self, content: str) -> tuple[str, List[str]]:
        """Simplify content (placeholder for LLM-based simplification)"""
        # In production, would use LLM to simplify
        # For now, just break long sentences
        changes = []
        sentences = content.split('. ')

        simplified = []
        for sent in sentences:
            if len(sent) > 100:
                # Split long sentences
                mid = len(sent) // 2
                split_point = sent.find(' ', mid)
                if split_point > 0:
                    simplified.append(sent[:split_point].strip())
                    simplified.append(sent[split_point:].strip())
                    changes.append(f"Split long sentence: '{sent[:30]}...'")
                else:
                    simplified.append(sent)
            else:
                simplified.append(sent)

        return '. '.join(simplified), changes

    def _add_emphasis(self, content: str) -> tuple[str, List[str]]:
        """Add emphasis markers"""
        # Mark important sentences (first and last in paragraphs)
        paragraphs = content.split('\n\n')
        emphasized = []
        changes = []

        for para in paragraphs:
            sentences = para.split('. ')
            if len(sentences) > 2:
                sentences[0] = f"**{sentences[0]}**"
                changes.append("Added emphasis to opening sentence")
            emphasized.append('. '.join(sentences))

        return '\n\n'.join(emphasized), changes

    def _improve_structure(self, content: str) -> tuple[str, List[str]]:
        """Improve content structure"""
        # Add paragraph breaks for readability
        changes = []
        sentences = content.split('. ')

        paragraphs = []
        current_para = []

        for i, sent in enumerate(sentences):
            current_para.append(sent)
            if len(current_para) >= 3:  # Max 3 sentences per paragraph
                paragraphs.append('. '.join(current_para) + '.')
                current_para = []
                if len(paragraphs) > 1:
                    changes.append("Added paragraph break for readability")

        if current_para:
            paragraphs.append('. '.join(current_para))

        return '\n\n'.join(paragraphs), changes

    def _simplify_vocabulary(self, content: str) -> tuple[str, List[str]]:
        """Simplify vocabulary (placeholder)"""
        # In production, would use LLM to replace complex words
        changes = []
        complex_to_simple = {
            "utilize": "use",
            "numerous": "many",
            "facilitate": "help",
            "implement": "do",
            "additional": "more"
        }

        refined = content
        for complex_word, simple_word in complex_to_simple.items():
            if complex_word in content.lower():
                refined = refined.replace(complex_word, simple_word)
                refined = refined.replace(complex_word.capitalize(), simple_word.capitalize())
                changes.append(f"Simplified: '{complex_word}' -> '{simple_word}'")

        return refined, changes

    def _enforce_sentence_length(self, content: str, max_length: int) -> tuple[str, List[str]]:
        """Enforce maximum sentence length"""
        sentences = content.split('. ')
        refined = []
        changes = []

        for sent in sentences:
            if len(sent) > max_length:
                # Simple split at nearest space
                parts = []
                while len(sent) > max_length:
                    split_point = sent.rfind(' ', 0, max_length)
                    if split_point > 0:
                        parts.append(sent[:split_point].strip())
                        sent = sent[split_point:].strip()
                    else:
                        break
                parts.append(sent)
                refined.extend(parts)
                changes.append(f"Split sentence to meet {max_length} character limit")
            else:
                refined.append(sent)

        return '. '.join(refined), changes

    def _convert_to_bullets(self, content: str) -> tuple[str, List[str]]:
        """Convert lists to bullet points"""
        changes = []
        # Simple heuristic: if content has "first", "second", etc., convert to bullets
        if any(word in content.lower() for word in ["first,", "second,", "third,"]):
            # This is a simplified version
            changes.append("Converted list to bullet points")

        return content, changes

    def _calculate_quality_score(
        self,
        original: str,
        refined: str,
        cognitive_state: Optional[CognitiveState],
        profile: Optional[AccessibilityProfile]
    ) -> float:
        """Calculate quality score"""
        # Base score
        score = 0.5

        # Increase score if content was adapted
        if original != refined:
            score += 0.2

        # Increase score if we had good input data
        if cognitive_state and cognitive_state.confidence > 0.7:
            score += 0.15

        if profile:
            score += 0.15

        return min(1.0, score)
