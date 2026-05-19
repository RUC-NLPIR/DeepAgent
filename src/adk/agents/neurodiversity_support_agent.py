"""
Neurodiversity Support Agent

Support for neurodivergent users of generative AI:
- Autism spectrum support (literal language, reduced ambiguity)
- ADHD support (focus aids, chunking, time management)
- Dyslexia support (font selection, spacing, formatting)
- Sensory processing support
- Executive function support
- Already includes alexithymia support (in bidirectional_reasoning.py)
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..utils.logger import get_logger


class NeurodiversitySupportAgent:
    """
    Agent for neurodiversity-aware content adaptation

    Supports various neurodivergent profiles:
    - Autism (literal communication, predictability)
    - ADHD (focus, attention management)
    - Dyslexia (reading support)
    - Sensory sensitivities
    - Executive function challenges
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Neurodiversity Support Agent"""
        self.config = config or {}
        self.logger = get_logger("system")

        # Dyslexia-friendly fonts
        self.dyslexia_fonts = [
            "OpenDyslexic",
            "Dyslexie",
            "Comic Sans MS",  # Surprisingly helpful
            "Arial",
            "Verdana"
        ]

        # Autism-friendly communication patterns
        self.ambiguous_terms = [
            "maybe", "possibly", "might", "could", "perhaps",
            "sort of", "kind of", "basically"
        ]

        self.logger.info("NeurodiversitySupportAgent initialized")

    async def adapt_for_autism(
        self,
        content: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Adapt content for autism spectrum users

        Focus areas:
        - Literal language (reduce idioms, metaphors)
        - Clear structure and predictability
        - Explicit instructions
        - Reduced ambiguity

        Args:
            content: Content to adapt
            preferences: User preferences

        Returns:
            Autism-friendly content and adaptations
        """
        adapted = content
        changes = []

        # 1. Remove/explain idioms and metaphors
        adapted, idiom_changes = self._clarify_idioms(adapted)
        changes.extend(idiom_changes)

        # 2. Make implicit information explicit
        adapted, explicit_changes = self._make_explicit(adapted)
        changes.extend(explicit_changes)

        # 3. Reduce ambiguity
        adapted, ambiguity_changes = self._reduce_ambiguity(adapted)
        changes.extend(ambiguity_changes)

        # 4. Add structure markers
        adapted, structure_changes = self._add_structure_markers(adapted)
        changes.extend(structure_changes)

        # 5. Simplify social language
        adapted, social_changes = self._simplify_social_language(adapted)
        changes.extend(social_changes)

        # Calculate improvement score
        improvement_score = min(len(changes) / 10.0, 1.0)

        return {
            "original_content": content,
            "adapted_content": adapted,
            "changes_made": changes,
            "improvement_score": round(improvement_score, 2),
            "autism_friendly_score": self._calculate_autism_friendliness(adapted),
            "recommendations": self._get_autism_recommendations(adapted),
            "timestamp": datetime.now().isoformat()
        }

    def _clarify_idioms(self, text: str) -> Tuple[str, List[str]]:
        """Identify and clarify idioms"""
        changes = []

        common_idioms = {
            "piece of cake": "easy task",
            "break the ice": "start a conversation",
            "hit the nail on the head": "be exactly right",
            "let the cat out of the bag": "reveal a secret",
            "it's raining cats and dogs": "it's raining heavily",
            "cost an arm and a leg": "be very expensive",
            "break a leg": "good luck",
            "under the weather": "feeling ill"
        }

        for idiom, literal in common_idioms.items():
            if idiom.lower() in text.lower():
                text = re.sub(
                    re.escape(idiom),
                    f"{idiom} (meaning: {literal})",
                    text,
                    flags=re.IGNORECASE
                )
                changes.append(f"Clarified idiom: '{idiom}' → '{literal}'")

        return text, changes

    def _make_explicit(self, text: str) -> Tuple[str, List[str]]:
        """Make implicit information explicit"""
        changes = []

        # Explicit sequencing
        if re.search(r'\bthen\b', text, re.IGNORECASE):
            # Already has sequencing
            pass
        else:
            # Check for implied sequence
            sentences = re.split(r'[.!?]', text)
            if len(sentences) > 2:
                changes.append("Consider adding explicit sequence markers (First, Then, Finally)")

        # Make pronouns more explicit (reduce ambiguity)
        pronoun_pattern = r'\b(it|this|that|these|those)\b'
        pronoun_count = len(re.findall(pronoun_pattern, text, re.IGNORECASE))

        if pronoun_count > 5:
            changes.append("Consider replacing some pronouns with specific nouns for clarity")

        return text, changes

    def _reduce_ambiguity(self, text: str) -> Tuple[str, List[str]]:
        """Reduce ambiguous language"""
        changes = []

        for term in self.ambiguous_terms:
            if f' {term} ' in text.lower():
                changes.append(f"Ambiguous term detected: '{term}' - consider more specific language")

        # Check for vague quantities
        vague_quantities = ["some", "few", "many", "several", "various"]
        for quantity in vague_quantities:
            if f' {quantity} ' in text.lower():
                changes.append(f"Vague quantity: '{quantity}' - consider specific numbers when possible")

        return text, changes

    def _add_structure_markers(self, text: str) -> Tuple[str, List[str]]:
        """Add explicit structure markers"""
        changes = []

        # Check if already has structure markers
        has_markers = bool(re.search(r'\b(first|second|third|finally|in conclusion)\b', text, re.IGNORECASE))

        if not has_markers and len(text.split('.')) > 3:
            changes.append("Consider adding structure markers: 'First', 'Second', 'Finally'")

        return text, changes

    def _simplify_social_language(self, text: str) -> Tuple[str, List[str]]:
        """Simplify social/emotional language"""
        changes = []

        # Detect emotional language that might be unclear
        emotional_terms = ["upset", "frustrated", "anxious", "excited"]

        for term in emotional_terms:
            if term in text.lower():
                changes.append(f"Emotional term '{term}' detected - consider adding explicit description")

        # Simplify polite ambiguity
        polite_ambiguous = {
            "would you mind": "please",
            "if you don't mind": "please",
            "could you perhaps": "please"
        }

        for polite, simple in polite_ambiguous.items():
            if polite in text.lower():
                text = re.sub(polite, simple, text, flags=re.IGNORECASE)
                changes.append(f"Simplified polite phrase: '{polite}' → '{simple}'")

        return text, changes

    def _calculate_autism_friendliness(self, text: str) -> float:
        """Calculate autism-friendliness score"""
        score = 1.0

        # Penalty for ambiguous language
        ambiguous_count = sum(1 for term in self.ambiguous_terms if term in text.lower())
        score -= min(ambiguous_count * 0.05, 0.3)

        # Penalty for idioms
        idiom_indicators = ["piece of", "break the", "under the"]
        idiom_count = sum(1 for idiom in idiom_indicators if idiom in text.lower())
        score -= min(idiom_count * 0.1, 0.3)

        # Bonus for explicit structure
        structure_markers = ["first", "second", "then", "finally", "in conclusion"]
        has_structure = any(marker in text.lower() for marker in structure_markers)
        if has_structure:
            score += 0.2

        return max(0.0, min(score, 1.0))

    def _get_autism_recommendations(self, text: str) -> List[str]:
        """Get autism-specific recommendations"""
        recommendations = []

        if len(text.split('.')) > 5:
            recommendations.append("Add clear headings to organize information")

        ambiguous_count = sum(1 for term in self.ambiguous_terms if term in text.lower())
        if ambiguous_count > 3:
            recommendations.append("Replace ambiguous terms with specific language")

        if "sarcasm" in text.lower() or "irony" in text.lower():
            recommendations.append("Avoid sarcasm and irony - use literal language")

        return recommendations

    async def adapt_for_adhd(
        self,
        content: str,
        chunk_size: int = 150,
        add_focus_aids: bool = True
    ) -> Dict[str, Any]:
        """
        Adapt content for ADHD users

        Focus areas:
        - Chunked information (reduce overwhelm)
        - Focus aids (key points highlighted)
        - Time estimates
        - Progress indicators
        - Reduced distractions

        Args:
            content: Content to adapt
            chunk_size: Words per chunk
            add_focus_aids: Add visual focus aids

        Returns:
            ADHD-friendly content
        """
        # Break into manageable chunks
        words = content.split()
        chunks = []
        current_chunk = []
        word_count = 0

        for word in words:
            current_chunk.append(word)
            word_count += 1

            # Try to break at sentence boundaries
            if word.endswith('.') and word_count >= chunk_size * 0.7:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                word_count = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        # Add focus aids
        if add_focus_aids:
            enhanced_chunks = []
            for i, chunk in enumerate(chunks, 1):
                # Extract key point (first sentence)
                key_point = chunk.split('.')[0] + '.'

                enhanced = f"**Key Point {i}:** {key_point}\n\n{chunk}"
                enhanced_chunks.append(enhanced)

            chunks = enhanced_chunks

        # Calculate time estimates
        total_words = len(words)
        reading_time = (total_words / 200) * 60  # 200 wpm, in seconds
        time_per_chunk = reading_time / len(chunks) if chunks else 0

        return {
            "original_content": content,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "words_per_chunk": chunk_size,
            "total_reading_time_seconds": round(reading_time, 1),
            "time_per_chunk_seconds": round(time_per_chunk, 1),
            "focus_aids_enabled": add_focus_aids,
            "recommendations": [
                "Take breaks between chunks",
                f"Estimated time: {int(reading_time // 60)} minutes {int(reading_time % 60)} seconds",
                "Use text-to-speech if available",
                "Minimize distractions while reading"
            ],
            "timestamp": datetime.now().isoformat()
        }

    async def adapt_for_dyslexia(
        self,
        content: str,
        font_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Adapt content for dyslexia

        Focus areas:
        - Font selection (dyslexia-friendly)
        - Spacing (increased letter and line spacing)
        - Avoid justified text
        - Shorter line lengths
        - Visual aids

        Args:
            content: Content to adapt
            font_preference: Preferred dyslexia-friendly font

        Returns:
            Dyslexia-friendly formatting recommendations
        """
        # Select font
        font = font_preference if font_preference in self.dyslexia_fonts else "OpenDyslexic"

        # Analyze content for dyslexia challenges
        challenges = []

        # Check for long words
        words = re.findall(r'\b\w+\b', content)
        long_words = [w for w in words if len(w) > 12]
        if len(long_words) > len(words) * 0.1:  # > 10% long words
            challenges.append("High proportion of long words")

        # Check for similar-looking words in proximity
        similar_pairs = [("form", "from"), ("quiet", "quite"), ("where", "were")]
        for word1, word2 in similar_pairs:
            if word1 in content.lower() and word2 in content.lower():
                challenges.append(f"Similar words present: '{word1}' and '{word2}'")

        # Formatting recommendations
        formatting = {
            "font_family": font,
            "font_size": "14-18pt",
            "line_spacing": 1.5,
            "letter_spacing": "0.12em",
            "paragraph_spacing": "2em",
            "text_alignment": "left (never justified)",
            "line_length": "60-70 characters",
            "background_color": "cream or light colored",
            "text_color": "dark gray (not pure black)"
        }

        # Content adaptations
        adapted_suggestions = []

        if long_words:
            adapted_suggestions.append("Consider breaking long words with hyphens")
            adapted_suggestions.append(f"Long words found: {', '.join(long_words[:5])}")

        # Suggest visual aids
        adapted_suggestions.append("Use bullet points instead of long paragraphs")
        adapted_suggestions.append("Add images or icons to support text")
        adapted_suggestions.append("Use color coding for different sections")

        return {
            "formatting_recommendations": formatting,
            "challenges_identified": challenges,
            "content_adaptations": adapted_suggestions,
            "dyslexia_friendliness_score": self._calculate_dyslexia_score(content),
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_dyslexia_score(self, text: str) -> float:
        """Calculate dyslexia-friendliness score"""
        score = 1.0

        # Check word length distribution
        words = re.findall(r'\b\w+\b', text)
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length > 6:
                score -= 0.2

            # Penalty for very long words
            very_long = sum(1 for w in words if len(w) > 12)
            score -= min(very_long / len(words), 0.3)

        # Check sentence length
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length > 20:
                score -= 0.2

        return max(0.0, min(score, 1.0))

    async def detect_sensory_overload_risk(
        self,
        content: str,
        has_animations: bool = False,
        has_flashing: bool = False,
        color_count: int = 0,
        sound_present: bool = False
    ) -> Dict[str, Any]:
        """
        Detect sensory overload risks

        Args:
            content: Content text
            has_animations: Whether content has animations
            has_flashing: Whether content has flashing elements
            color_count: Number of distinct colors used
            sound_present: Whether sound is present

        Returns:
            Sensory overload risk assessment
        """
        risk_score = 0.0
        risks = []
        mitigation_strategies = []

        # Visual complexity
        word_count = len(content.split())
        if word_count > 500:
            risk_score += 0.2
            risks.append("High text density")
            mitigation_strategies.append("Break into smaller sections")

        # Animation risk
        if has_animations:
            risk_score += 0.3
            risks.append("Animated content present")
            mitigation_strategies.append("Provide option to disable animations")

        # Flashing content (high risk)
        if has_flashing:
            risk_score += 0.5
            risks.append("⚠️  FLASHING CONTENT - Seizure risk")
            mitigation_strategies.append("URGENT: Remove flashing or add warning")

        # Color complexity
        if color_count > 8:
            risk_score += 0.2
            risks.append("High color complexity")
            mitigation_strategies.append("Reduce color palette")

        # Auditory
        if sound_present:
            risk_score += 0.2
            risks.append("Audio present")
            mitigation_strategies.append("Provide volume controls and mute option")

        risk_score = min(risk_score, 1.0)

        risk_level = (
            "low" if risk_score < 0.3 else
            "medium" if risk_score < 0.6 else
            "high"
        )

        return {
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "risks_identified": risks,
            "mitigation_strategies": mitigation_strategies,
            "wcag_compliant": not has_flashing and color_count <= 8,
            "timestamp": datetime.now().isoformat()
        }

    async def provide_executive_function_support(
        self,
        task_description: str
    ) -> Dict[str, Any]:
        """
        Provide executive function support for tasks

        Breaks down tasks, adds time estimates, creates checklists

        Args:
            task_description: Description of task to support

        Returns:
            Executive function support materials
        """
        # Break task into steps
        sentences = re.split(r'[.!?]', task_description)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Create checklist
        checklist = []
        for i, sentence in enumerate(sentences, 1):
            # Make each sentence actionable
            if not sentence.lower().startswith(('step', 'first', 'then', 'next', 'finally')):
                sentence = f"Step {i}: {sentence}"

            checklist.append({
                "step_number": i,
                "instruction": sentence,
                "completed": False,
                "estimated_time_minutes": 5  # Default estimate
            })

        # Add time buffer
        total_time = len(checklist) * 5
        buffered_time = int(total_time * 1.5)  # 50% buffer

        return {
            "task_description": task_description,
            "checklist": checklist,
            "total_steps": len(checklist),
            "estimated_time_minutes": total_time,
            "recommended_time_with_buffer": buffered_time,
            "executive_function_aids": [
                "Set timer for each step",
                "Take 2-minute break every 15 minutes",
                "Use visual progress tracker",
                "Minimize distractions during task",
                "Review checklist before starting"
            ],
            "timestamp": datetime.now().isoformat()
        }
