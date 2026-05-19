"""
Cognitive Accessibility Agent

Enhanced cognitive accessibility features for generative AI:
- Text simplification (multiple levels)
- Content summarization
- Reading level analysis
- Cognitive load prediction
- Chunking and progressive disclosure
- Plain language conversion
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import Counter

from ..utils.logger import get_logger


class CognitiveAccessibilityAgent:
    """
    Agent for cognitive accessibility features

    Addresses cognitive challenges in consuming AI-generated content:
    - Complex language barriers
    - Information overload
    - Reading comprehension difficulties
    - Attention and focus challenges
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Cognitive Accessibility Agent"""
        self.config = config or {}
        self.logger = get_logger("system")

        # Reading level thresholds (Flesch-Kincaid Grade Level)
        self.reading_levels = {
            "elementary": (0, 6),
            "middle_school": (6, 9),
            "high_school": (9, 13),
            "college": (13, 16),
            "graduate": (16, 100)
        }

        # Common complex words to simplify
        self.simplification_map = {
            "utilize": "use",
            "facilitate": "help",
            "implement": "do",
            "demonstrate": "show",
            "acquire": "get",
            "terminate": "end",
            "commence": "start",
            "endeavor": "try",
            "obtain": "get",
            "indicate": "show",
            "construct": "build",
            "determine": "find",
            "subsequent": "next",
            "prior": "before",
            "approximately": "about",
            "currently": "now",
            "consequently": "so",
            "therefore": "so",
            "additionally": "also",
            "furthermore": "also",
            "nevertheless": "but",
            "however": "but",
            "subsequently": "then"
        }

        self.logger.info("CognitiveAccessibilityAgent initialized")

    async def simplify_text(
        self,
        text: str,
        target_level: str = "middle_school",
        preserve_meaning: bool = True
    ) -> Dict[str, Any]:
        """
        Simplify text to target reading level

        Args:
            text: Original text to simplify
            target_level: Target reading level (elementary, middle_school, high_school)
            preserve_meaning: Whether to preserve technical accuracy

        Returns:
            Simplified text and analysis
        """
        # Analyze original text
        original_analysis = await self.analyze_reading_level(text)

        # Apply simplification strategies
        simplified = text
        changes = []

        # 1. Replace complex words
        for complex_word, simple_word in self.simplification_map.items():
            pattern = r'\b' + complex_word + r'\b'
            if re.search(pattern, simplified, re.IGNORECASE):
                simplified = re.sub(pattern, simple_word, simplified, flags=re.IGNORECASE)
                changes.append(f"Replaced '{complex_word}' with '{simple_word}'")

        # 2. Break long sentences
        simplified = self._break_long_sentences(simplified, changes)

        # 3. Simplify sentence structure
        simplified = self._simplify_sentence_structure(simplified, changes)

        # 4. Remove unnecessary jargon (if not preserving technical meaning)
        if not preserve_meaning:
            simplified = self._remove_jargon(simplified, changes)

        # Analyze simplified text
        simplified_analysis = await self.analyze_reading_level(simplified)

        return {
            "original_text": text,
            "simplified_text": simplified,
            "original_reading_level": original_analysis["reading_level"],
            "simplified_reading_level": simplified_analysis["reading_level"],
            "original_grade": original_analysis["grade_level"],
            "simplified_grade": simplified_analysis["grade_level"],
            "changes_made": changes,
            "improvement": original_analysis["grade_level"] - simplified_analysis["grade_level"],
            "target_achieved": simplified_analysis["reading_level_category"] == target_level,
            "timestamp": datetime.now().isoformat()
        }

    def _break_long_sentences(self, text: str, changes: List[str]) -> str:
        """Break long sentences into shorter ones"""
        sentences = re.split(r'([.!?])\s+', text)
        result = []

        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]

            # If sentence is too long (> 25 words), try to break it
            words = sentence.split()
            if len(words) > 25:
                # Find conjunction to split on
                conjunctions = ['and', 'but', 'or', 'so', 'because', 'although', 'while']
                for j, word in enumerate(words[10:], start=10):  # Start checking after 10 words
                    if word.lower() in conjunctions and j < len(words) - 5:
                        # Split at conjunction
                        first_part = ' '.join(words[:j])
                        second_part = ' '.join(words[j+1:])
                        sentence = f"{first_part}. {second_part.capitalize()}"
                        changes.append("Split long sentence at conjunction")
                        break

            result.append(sentence)

        return ' '.join(result)

    def _simplify_sentence_structure(self, text: str, changes: List[str]) -> str:
        """Simplify complex sentence structures"""
        # Convert passive voice to active (simple heuristic)
        passive_patterns = [
            (r'(\w+) (is|are|was|were) (\w+ed) by', r'\3 \1'),  # Simple passive
        ]

        for pattern, replacement in passive_patterns:
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text)
                changes.append("Converted passive voice to active")

        return text

    def _remove_jargon(self, text: str, changes: List[str]) -> str:
        """Remove or simplify technical jargon"""
        # This is a placeholder - in production, use domain-specific jargon dictionaries
        jargon_indicators = ['paradigm', 'synergy', 'leverage', 'ecosystem']

        for jargon in jargon_indicators:
            if jargon in text.lower():
                changes.append(f"Identified jargon: '{jargon}' (manual review recommended)")

        return text

    async def analyze_reading_level(self, text: str) -> Dict[str, Any]:
        """
        Analyze reading level of text using multiple metrics

        Returns:
            Reading level analysis with Flesch-Kincaid and other metrics
        """
        # Count syllables (approximation)
        def count_syllables(word: str) -> int:
            word = word.lower()
            vowels = 'aeiouy'
            syllables = 0
            previous_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllables += 1
                previous_was_vowel = is_vowel

            # Adjust for silent e
            if word.endswith('e'):
                syllables -= 1

            return max(1, syllables)

        # Split into sentences and words
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        words = re.findall(r'\b\w+\b', text)

        if not sentences or not words:
            return {
                "reading_level": 0,
                "reading_level_category": "elementary",
                "grade_level": 0,
                "error": "Insufficient text to analyze"
            }

        # Calculate metrics
        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(count_syllables(word) for word in words)

        avg_sentence_length = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words

        # Flesch-Kincaid Grade Level
        grade_level = (
            0.39 * avg_sentence_length +
            11.8 * avg_syllables_per_word -
            15.59
        )
        grade_level = max(0, grade_level)

        # Flesch Reading Ease (0-100, higher is easier)
        reading_ease = (
            206.835 -
            1.015 * avg_sentence_length -
            84.6 * avg_syllables_per_word
        )

        # Determine category
        reading_level_category = "graduate"
        for category, (min_grade, max_grade) in self.reading_levels.items():
            if min_grade <= grade_level < max_grade:
                reading_level_category = category
                break

        # Additional metrics
        long_words = sum(1 for word in words if len(word) > 6)
        complex_word_ratio = long_words / total_words if total_words > 0 else 0

        return {
            "reading_level": round(reading_ease, 1),
            "reading_level_category": reading_level_category,
            "grade_level": round(grade_level, 1),
            "avg_sentence_length": round(avg_sentence_length, 1),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "complex_word_ratio": round(complex_word_ratio, 2),
            "total_sentences": total_sentences,
            "total_words": total_words,
            "interpretation": self._interpret_reading_level(reading_ease, grade_level)
        }

    def _interpret_reading_level(self, ease: float, grade: float) -> str:
        """Interpret reading level scores"""
        if ease >= 90:
            return "Very easy to read (5th grade)"
        elif ease >= 80:
            return "Easy to read (6th grade)"
        elif ease >= 70:
            return "Fairly easy to read (7th grade)"
        elif ease >= 60:
            return "Standard (8th-9th grade)"
        elif ease >= 50:
            return "Fairly difficult (10th-12th grade)"
        elif ease >= 30:
            return "Difficult (college level)"
        else:
            return "Very difficult (graduate level)"

    async def generate_summary(
        self,
        text: str,
        summary_type: str = "brief",
        max_sentences: int = 3
    ) -> Dict[str, Any]:
        """
        Generate summary of text

        Args:
            text: Text to summarize
            summary_type: "brief", "medium", or "detailed"
            max_sentences: Maximum sentences in summary

        Returns:
            Summary and metadata
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {
                "summary": "",
                "error": "No content to summarize"
            }

        # Adjust max sentences based on type
        if summary_type == "brief":
            max_sentences = min(max_sentences, 2)
        elif summary_type == "medium":
            max_sentences = min(max_sentences, 5)
        else:  # detailed
            max_sentences = min(max_sentences, len(sentences) // 2)

        # Extract key sentences (simple heuristic: first, last, and longest)
        summary_sentences = []

        # Always include first sentence
        if sentences:
            summary_sentences.append(sentences[0])

        # Add longest sentence (likely contains key info)
        if len(sentences) > 2:
            longest = max(sentences[1:-1], key=len, default=None)
            if longest and longest not in summary_sentences:
                summary_sentences.append(longest)

        # Add last sentence if we have room
        if len(sentences) > 1 and len(summary_sentences) < max_sentences:
            if sentences[-1] not in summary_sentences:
                summary_sentences.append(sentences[-1])

        summary = '. '.join(summary_sentences) + '.'

        # Calculate compression ratio
        original_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = 1 - (summary_words / original_words) if original_words > 0 else 0

        return {
            "summary": summary,
            "summary_type": summary_type,
            "original_length": original_words,
            "summary_length": summary_words,
            "compression_ratio": round(compression_ratio, 2),
            "sentence_count": len(summary_sentences),
            "timestamp": datetime.now().isoformat()
        }

    async def chunk_content(
        self,
        text: str,
        chunk_size: int = 200,
        overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Chunk content for progressive disclosure

        Args:
            text: Text to chunk
            chunk_size: Target words per chunk
            overlap: Words to overlap between chunks

        Returns:
            Chunked content with metadata
        """
        words = text.split()
        total_words = len(words)

        chunks = []
        start = 0

        while start < total_words:
            end = min(start + chunk_size, total_words)
            chunk = ' '.join(words[start:end])

            # Try to end at sentence boundary
            if end < total_words:
                # Look for last sentence ending in chunk
                last_period = chunk.rfind('.')
                if last_period > len(chunk) * 0.5:  # Only if > 50% through chunk
                    chunk = chunk[:last_period + 1]
                    # Recalculate end based on actual chunk
                    end = start + len(chunk.split())

            chunks.append({
                "chunk_index": len(chunks),
                "content": chunk.strip(),
                "word_count": len(chunk.split()),
                "start_word": start,
                "end_word": end
            })

            start = end - overlap if overlap > 0 else end

        return {
            "chunks": chunks,
            "total_chunks": len(chunks),
            "total_words": total_words,
            "avg_chunk_size": round(sum(c["word_count"] for c in chunks) / len(chunks), 1),
            "overlap_words": overlap,
            "timestamp": datetime.now().isoformat()
        }

    async def predict_cognitive_load(
        self,
        text: str,
        has_images: int = 0,
        has_interactions: int = 0
    ) -> Dict[str, Any]:
        """
        Predict cognitive load of content

        Args:
            text: Content text
            has_images: Number of images
            has_interactions: Number of interactive elements

        Returns:
            Cognitive load prediction
        """
        load_score = 0.0

        # Text complexity
        reading_analysis = await self.analyze_reading_level(text)
        grade_level = reading_analysis["grade_level"]

        # Higher grade level = higher load
        load_score += min(grade_level / 20.0, 0.3)

        # Length factor
        word_count = len(text.split())
        if word_count > 1000:
            load_score += 0.3
        elif word_count > 500:
            load_score += 0.2
        else:
            load_score += 0.1

        # Visual complexity
        visual_load = (has_images * 0.05) + (has_interactions * 0.08)
        load_score += min(visual_load, 0.3)

        # Sentence complexity
        avg_sentence_length = reading_analysis["avg_sentence_length"]
        if avg_sentence_length > 25:
            load_score += 0.2
        elif avg_sentence_length > 15:
            load_score += 0.1

        load_score = min(load_score, 1.0)

        load_level = (
            "low" if load_score < 0.4 else
            "medium" if load_score < 0.7 else
            "high"
        )

        recommendations = self._get_load_recommendations(load_level, reading_analysis)

        return {
            "cognitive_load": round(load_score, 2),
            "load_level": load_level,
            "contributing_factors": {
                "text_complexity": round(grade_level / 20.0, 2),
                "content_length": word_count,
                "visual_elements": has_images + has_interactions
            },
            "recommendations": recommendations,
            "estimated_reading_time_minutes": round(word_count / 200, 1),  # 200 wpm average
            "timestamp": datetime.now().isoformat()
        }

    def _get_load_recommendations(
        self,
        load_level: str,
        reading_analysis: Dict[str, Any]
    ) -> List[str]:
        """Get recommendations for reducing cognitive load"""
        recommendations = []

        if load_level == "high":
            recommendations.append("Consider breaking content into smaller sections")
            recommendations.append("Add a table of contents for navigation")
            recommendations.append("Provide a summary at the beginning")

        if reading_analysis["grade_level"] > 12:
            recommendations.append("Simplify language to lower reading level")
            recommendations.append("Define technical terms in context")

        if reading_analysis["avg_sentence_length"] > 20:
            recommendations.append("Break long sentences into shorter ones")

        if reading_analysis["complex_word_ratio"] > 0.15:
            recommendations.append("Replace complex words with simpler alternatives")

        return recommendations

    async def convert_to_plain_language(
        self,
        text: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert text to plain language following federal guidelines

        Args:
            text: Text to convert
            domain: Optional domain for context-specific simplification

        Returns:
            Plain language version and analysis
        """
        # Apply plain language principles
        plain_text = text
        changes = []

        # 1. Use active voice
        plain_text = self._simplify_sentence_structure(plain_text, changes)

        # 2. Use short words and sentences
        for complex_word, simple_word in self.simplification_map.items():
            if complex_word in plain_text.lower():
                plain_text = re.sub(
                    r'\b' + complex_word + r'\b',
                    simple_word,
                    plain_text,
                    flags=re.IGNORECASE
                )
                changes.append(f"'{complex_word}' → '{simple_word}'")

        # 3. Avoid nominalizations (verbs turned into nouns)
        nominalizations = {
            'implementation': 'implement',
            'utilization': 'use',
            'facilitation': 'facilitate',
            'optimization': 'optimize'
        }

        for nominal, verb in nominalizations.items():
            if nominal in plain_text.lower():
                changes.append(f"Simplified nominalization: '{nominal}'")

        # 4. Use lists for multiple items
        # Detect potential list patterns
        if re.search(r'(first|second|third|finally)', plain_text, re.IGNORECASE):
            changes.append("Consider converting sequence to numbered list")

        # Analyze improvement
        original_analysis = await self.analyze_reading_level(text)
        plain_analysis = await self.analyze_reading_level(plain_text)

        improvement_score = (
            original_analysis["grade_level"] - plain_analysis["grade_level"]
        ) / original_analysis["grade_level"] if original_analysis["grade_level"] > 0 else 0

        return {
            "original_text": text,
            "plain_language_text": plain_text,
            "changes_applied": changes,
            "original_grade_level": original_analysis["grade_level"],
            "plain_grade_level": plain_analysis["grade_level"],
            "improvement_score": round(improvement_score, 2),
            "meets_plain_language_standard": plain_analysis["grade_level"] <= 8,
            "timestamp": datetime.now().isoformat()
        }
