"""
Multimodal Output Agent

Generates accessible output in multiple modalities:
- Text-to-speech preparation
- Audio description generation
- Structured data formats (JSON, XML, etc.)
- Visual representations of text
- Tactile-friendly formatting
- Alternative format generation
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

from ..utils.logger import get_logger


class MultimodalOutputAgent:
    """
    Agent for generating multimodal accessible outputs

    Converts content into multiple accessible formats:
    - Screen reader optimized
    - Audio-ready text
    - Structured data
    - Print-friendly formats
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Multimodal Output Agent"""
        self.config = config or {}
        self.logger = get_logger("system")

        # SSML (Speech Synthesis Markup Language) tags for TTS
        self.prosody_settings = {
            "slow": {"rate": "slow", "pitch": "medium"},
            "medium": {"rate": "medium", "pitch": "medium"},
            "fast": {"rate": "fast", "pitch": "medium"}
        }

        self.logger.info("MultimodalOutputAgent initialized")

    async def prepare_for_tts(
        self,
        text: str,
        speech_rate: str = "medium",
        add_pauses: bool = True,
        pronunciation_hints: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare text for text-to-speech synthesis

        Args:
            text: Text to prepare
            speech_rate: "slow", "medium", or "fast"
            add_pauses: Add natural pauses at punctuation
            pronunciation_hints: Dict of word: pronunciation

        Returns:
            TTS-ready text with SSML markup and metadata
        """
        tts_text = text

        # Expand abbreviations for better TTS
        tts_text = self._expand_abbreviations(tts_text)

        # Add pauses at punctuation
        if add_pauses:
            tts_text = self._add_speech_pauses(tts_text)

        # Add pronunciation hints
        if pronunciation_hints:
            for word, pronunciation in pronunciation_hints.items():
                # In production, use proper SSML phoneme tags
                tts_text = re.sub(
                    r'\b' + re.escape(word) + r'\b',
                    f'{word} ({pronunciation})',
                    tts_text,
                    flags=re.IGNORECASE
                )

        # Generate SSML (simplified version)
        prosody = self.prosody_settings.get(speech_rate, self.prosody_settings["medium"])

        ssml = f'''<speak>
  <prosody rate="{prosody['rate']}" pitch="{prosody['pitch']}">
    {tts_text}
  </prosody>
</speak>'''

        # Estimate speech duration (rough approximation: 150 words per minute at medium)
        word_count = len(text.split())
        rates = {"slow": 100, "medium": 150, "fast": 200}
        estimated_duration_seconds = (word_count / rates[speech_rate]) * 60

        return {
            "original_text": text,
            "tts_text": tts_text,
            "ssml": ssml,
            "speech_rate": speech_rate,
            "estimated_duration_seconds": round(estimated_duration_seconds, 1),
            "word_count": word_count,
            "timestamp": datetime.now().isoformat()
        }

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations for TTS"""
        abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\betc\.': 'et cetera',
            r'\be\.g\.': 'for example',
            r'\bi\.e\.': 'that is',
            r'\bvs\.': 'versus',
            r'\bUSA\b': 'U S A',
            r'\bAI\b': 'A I',
            r'\bAPI\b': 'A P I',
            r'\bURL\b': 'U R L',
            r'\bHTML\b': 'H T M L',
            r'\bCSS\b': 'C S S',
        }

        for abbrev, expansion in abbreviations.items():
            text = re.sub(abbrev, expansion, text)

        return text

    def _add_speech_pauses(self, text: str) -> str:
        """Add pauses at punctuation for natural speech"""
        # Add short pause after commas
        text = text.replace(',', ', <break time="300ms"/>')

        # Add medium pause after periods
        text = text.replace('.', '. <break time="500ms"/>')

        # Add longer pause after paragraphs
        text = text.replace('\n\n', ' <break time="800ms"/> ')

        return text

    async def generate_audio_description(
        self,
        visual_content: str,
        context: Optional[str] = None,
        detail_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Generate audio description for visual content

        Args:
            visual_content: Description of visual content
            context: Context where content appears
            detail_level: "brief", "standard", or "extended"

        Returns:
            Audio description optimized for non-visual consumption
        """
        # Start with visual content
        audio_desc = visual_content

        # Add context if provided
        if context:
            audio_desc = f"In the context of {context}, {audio_desc.lower()}"

        # Adjust detail based on level
        if detail_level == "brief":
            # Extract key information only
            sentences = re.split(r'[.!?]', audio_desc)
            audio_desc = sentences[0] + '.' if sentences else audio_desc

        elif detail_level == "extended":
            # Add spatial and temporal details
            audio_desc = self._add_extended_details(audio_desc)

        # Make it more conversational for audio
        audio_desc = self._conversationalize(audio_desc)

        # Estimate reading time (140 words per minute for audio description)
        word_count = len(audio_desc.split())
        estimated_duration = (word_count / 140) * 60

        return {
            "audio_description": audio_desc,
            "detail_level": detail_level,
            "word_count": word_count,
            "estimated_duration_seconds": round(estimated_duration, 1),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

    def _add_extended_details(self, description: str) -> str:
        """Add extended details for comprehensive audio description"""
        # In production, this would use more sophisticated analysis
        # For now, add basic spatial markers
        if "image" in description.lower() or "shows" in description.lower():
            description += " The layout is arranged from top to bottom."

        return description

    def _conversationalize(self, text: str) -> str:
        """Make text more conversational for audio"""
        # Replace some formal constructs
        replacements = {
            "is depicted": "shows",
            "is illustrated": "shows",
            "demonstrates": "shows",
            "comprises": "includes",
        }

        for formal, casual in replacements.items():
            text = re.sub(formal, casual, text, flags=re.IGNORECASE)

        return text

    async def generate_structured_output(
        self,
        content: str,
        format_type: str = "json",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Generate structured data output

        Args:
            content: Content to structure
            format_type: "json", "xml", "yaml", "markdown"
            include_metadata: Include accessibility metadata

        Returns:
            Structured format output
        """
        # Parse content into structure
        structure = self._parse_content_structure(content)

        # Generate in requested format
        if format_type == "json":
            structured = self._to_json(structure, include_metadata)
        elif format_type == "xml":
            structured = self._to_xml(structure, include_metadata)
        elif format_type == "markdown":
            structured = self._to_markdown(structure, include_metadata)
        else:  # yaml
            structured = self._to_yaml(structure, include_metadata)

        return {
            "format": format_type,
            "structured_output": structured,
            "include_metadata": include_metadata,
            "accessibility_features": self._get_format_accessibility_features(format_type),
            "timestamp": datetime.now().isoformat()
        }

    def _parse_content_structure(self, content: str) -> Dict[str, Any]:
        """Parse content into structured components"""
        # Simple parsing - in production, use more sophisticated NLP
        paragraphs = content.split('\n\n')

        structure = {
            "content": content,
            "paragraphs": [p.strip() for p in paragraphs if p.strip()],
            "word_count": len(content.split()),
            "character_count": len(content)
        }

        # Detect headings (lines ending with specific patterns)
        headings = []
        body = []

        for para in structure["paragraphs"]:
            # Simple heuristic: short lines might be headings
            if len(para.split()) <= 10 and not para.endswith('.'):
                headings.append(para)
            else:
                body.append(para)

        structure["headings"] = headings
        structure["body_paragraphs"] = body

        return structure

    def _to_json(self, structure: Dict[str, Any], include_metadata: bool) -> str:
        """Convert to JSON format"""
        output = {
            "content": structure["content"],
            "sections": structure["body_paragraphs"]
        }

        if include_metadata:
            output["metadata"] = {
                "word_count": structure["word_count"],
                "paragraph_count": len(structure["paragraphs"]),
                "accessibility": {
                    "screen_reader_friendly": True,
                    "semantic_structure": True
                }
            }

        return json.dumps(output, indent=2, ensure_ascii=False)

    def _to_xml(self, structure: Dict[str, Any], include_metadata: bool) -> str:
        """Convert to XML format"""
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append('<document>')

        if include_metadata:
            xml_parts.append('  <metadata>')
            xml_parts.append(f'    <wordCount>{structure["word_count"]}</wordCount>')
            xml_parts.append(f'    <paragraphCount>{len(structure["paragraphs"])}</paragraphCount>')
            xml_parts.append('  </metadata>')

        xml_parts.append('  <content>')
        for i, para in enumerate(structure["body_paragraphs"], 1):
            xml_parts.append(f'    <paragraph id="{i}">{self._escape_xml(para)}</paragraph>')
        xml_parts.append('  </content>')
        xml_parts.append('</document>')

        return '\n'.join(xml_parts)

    def _to_markdown(self, structure: Dict[str, Any], include_metadata: bool) -> str:
        """Convert to Markdown format"""
        md_parts = []

        if structure["headings"]:
            md_parts.append(f'# {structure["headings"][0]}\n')

        for para in structure["body_paragraphs"]:
            md_parts.append(para)
            md_parts.append('')  # Empty line between paragraphs

        if include_metadata:
            md_parts.append('---')
            md_parts.append('## Metadata')
            md_parts.append(f'- Word Count: {structure["word_count"]}')
            md_parts.append(f'- Paragraphs: {len(structure["paragraphs"])}')

        return '\n'.join(md_parts)

    def _to_yaml(self, structure: Dict[str, Any], include_metadata: bool) -> str:
        """Convert to YAML format"""
        yaml_parts = []

        if include_metadata:
            yaml_parts.append('metadata:')
            yaml_parts.append(f'  word_count: {structure["word_count"]}')
            yaml_parts.append(f'  paragraph_count: {len(structure["paragraphs"])}')
            yaml_parts.append('')

        yaml_parts.append('content:')
        for i, para in enumerate(structure["body_paragraphs"]):
            yaml_parts.append(f'  - paragraph_{i + 1}: |')
            for line in para.split('\n'):
                yaml_parts.append(f'      {line}')

        return '\n'.join(yaml_parts)

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))

    def _get_format_accessibility_features(self, format_type: str) -> List[str]:
        """Get accessibility features of each format"""
        features = {
            "json": [
                "Machine-readable",
                "Screen reader compatible",
                "Programmable accessibility"
            ],
            "xml": [
                "Semantic structure",
                "Assistive technology support",
                "Metadata support"
            ],
            "markdown": [
                "Human-readable",
                "Screen reader friendly",
                "Easy to navigate"
            ],
            "yaml": [
                "Human-readable",
                "Hierarchical structure",
                "Configuration-friendly"
            ]
        }

        return features.get(format_type, [])

    async def generate_print_friendly(
        self,
        content: str,
        font_size: int = 12,
        line_spacing: float = 1.5,
        page_width: int = 80
    ) -> Dict[str, Any]:
        """
        Generate print-friendly format

        Args:
            content: Content to format
            font_size: Font size in points
            line_spacing: Line spacing multiplier
            page_width: Characters per line

        Returns:
            Print-optimized content
        """
        # Word wrap to page width
        words = content.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length > page_width:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += word_length

        if current_line:
            lines.append(' '.join(current_line))

        # Add spacing between lines
        spacing_char = '\n' * int(line_spacing)
        formatted = spacing_char.join(lines)

        # Estimate page count (assuming 60 lines per page)
        lines_per_page = int(60 / line_spacing)
        page_count = max(1, len(lines) // lines_per_page)

        return {
            "formatted_content": formatted,
            "font_size": font_size,
            "line_spacing": line_spacing,
            "characters_per_line": page_width,
            "total_lines": len(lines),
            "estimated_pages": page_count,
            "dyslexia_friendly": font_size >= 12 and line_spacing >= 1.5,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_braille_ready(
        self,
        text: str,
        grade: int = 2
    ) -> Dict[str, Any]:
        """
        Prepare text for Braille translation

        Args:
            text: Text to prepare
            grade: Braille grade (1 or 2)

        Returns:
            Braille-ready text
        """
        # Prepare text for Braille translation
        braille_ready = text

        # Expand contractions for Grade 1
        if grade == 1:
            braille_ready = self._expand_contractions(braille_ready)

        # Remove extra whitespace
        braille_ready = ' '.join(braille_ready.split())

        # Estimate Braille cell count (roughly 1 cell per character for Grade 1)
        cell_count_estimate = len(braille_ready) if grade == 1 else int(len(braille_ready) * 0.6)

        return {
            "braille_ready_text": braille_ready,
            "grade": grade,
            "original_characters": len(text),
            "estimated_braille_cells": cell_count_estimate,
            "note": "This is prepared for Braille translation software",
            "timestamp": datetime.now().isoformat()
        }

    def _expand_contractions(self, text: str) -> str:
        """Expand contractions for Grade 1 Braille"""
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "it's": "it is",
            "I'm": "I am",
            "you're": "you are",
            "they're": "they are",
            "we're": "we are"
        }

        for contraction, expanded in contractions.items():
            text = re.sub(
                r'\b' + re.escape(contraction) + r'\b',
                expanded,
                text,
                flags=re.IGNORECASE
            )

        return text
