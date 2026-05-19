"""
Visual Accessibility Agent

Handles visual accessibility features for generative AI:
- Alt text generation for images
- Screen reader optimization
- Color blindness detection and correction
- Visual complexity analysis
- Image description quality assessment
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

from ..utils.schemas import AccessibilityAdaptation
from ..utils.logger import get_logger


class VisualAccessibilityAgent:
    """
    Agent for visual accessibility features

    Addresses key visual accessibility challenges in generative AI:
    - Image content needs descriptive alt text
    - Color combinations must be accessible
    - Visual complexity affects comprehension
    - Screen reader compatibility
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Visual Accessibility Agent"""
        self.config = config or {}
        self.logger = get_logger("system")

        # Color blindness simulation matrices
        self.colorblind_types = {
            "protanopia": "red-blind",
            "deuteranopia": "green-blind",
            "tritanopia": "blue-blind",
            "achromatopsia": "total color blindness"
        }

        self.logger.info("VisualAccessibilityAgent initialized")

    async def generate_alt_text(
        self,
        image_description: str,
        context: Optional[str] = None,
        detail_level: str = "medium"
    ) -> Dict[str, Any]:
        """
        Generate accessible alt text for images

        Args:
            image_description: Description or analysis of the image
            context: Context where image is used
            detail_level: "brief", "medium", or "detailed"

        Returns:
            Dictionary with alt text and metadata
        """
        # Analyze image description quality
        quality_score = self._assess_description_quality(image_description)

        # Generate alt text based on detail level
        if detail_level == "brief":
            alt_text = self._generate_brief_alt(image_description)
        elif detail_level == "detailed":
            alt_text = self._generate_detailed_alt(image_description, context)
        else:  # medium
            alt_text = self._generate_medium_alt(image_description)

        # Add context if provided
        if context and detail_level != "brief":
            alt_text = f"{alt_text} (Context: {context})"

        return {
            "alt_text": alt_text,
            "detail_level": detail_level,
            "quality_score": quality_score,
            "character_count": len(alt_text),
            "meets_wcag": len(alt_text) > 0 and len(alt_text) < 250,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_brief_alt(self, description: str) -> str:
        """Generate brief alt text (< 50 chars)"""
        # Extract key objects/subjects
        words = description.split()
        if len(words) <= 6:
            return description
        return " ".join(words[:6]) + "..."

    def _generate_medium_alt(self, description: str) -> str:
        """Generate medium alt text (< 125 chars)"""
        if len(description) <= 125:
            return description

        # Truncate intelligently at sentence boundary
        sentences = re.split(r'[.!?]', description)
        alt_text = sentences[0]

        for sentence in sentences[1:]:
            if len(alt_text) + len(sentence) < 125:
                alt_text += ". " + sentence.strip()
            else:
                break

        return alt_text.strip()

    def _generate_detailed_alt(self, description: str, context: Optional[str]) -> str:
        """Generate detailed alt text (< 250 chars)"""
        # Combine description with context for detailed version
        base = description[:200] if len(description) > 200 else description

        if context and len(base) + len(context) < 240:
            return f"{base} Located in: {context}"

        return base

    def _assess_description_quality(self, description: str) -> float:
        """
        Assess quality of image description

        Returns quality score 0.0-1.0 based on:
        - Length (not too short, not too long)
        - Specificity (contains concrete nouns)
        - Clarity (readable language)
        """
        score = 0.0

        # Length check
        length = len(description)
        if 20 <= length <= 200:
            score += 0.3
        elif length > 200:
            score += 0.2
        else:
            score += 0.1

        # Specificity (has nouns/objects)
        common_objects = ['person', 'people', 'image', 'photo', 'chart', 'graph',
                         'diagram', 'table', 'text', 'button', 'icon']
        has_objects = any(obj in description.lower() for obj in common_objects)
        if has_objects:
            score += 0.3

        # Clarity (no placeholder text)
        placeholder_text = ['[insert', 'tbd', 'todo', 'placeholder', 'xxx']
        has_placeholder = any(ph in description.lower() for ph in placeholder_text)
        if not has_placeholder:
            score += 0.2

        # Actionable (describes what's happening)
        action_words = ['showing', 'displaying', 'depicting', 'containing', 'featuring']
        has_action = any(word in description.lower() for word in action_words)
        if has_action:
            score += 0.2

        return min(score, 1.0)

    async def check_color_contrast(
        self,
        foreground_color: Tuple[int, int, int],
        background_color: Tuple[int, int, int],
        text_size: float = 14.0
    ) -> Dict[str, Any]:
        """
        Check color contrast ratio for WCAG compliance

        Args:
            foreground_color: RGB tuple for text color
            background_color: RGB tuple for background
            text_size: Font size in points

        Returns:
            Dictionary with contrast analysis
        """
        # Calculate relative luminance
        def relative_luminance(rgb: Tuple[int, int, int]) -> float:
            r, g, b = [x / 255.0 for x in rgb]
            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        # Calculate contrast ratio
        l1 = relative_luminance(foreground_color)
        l2 = relative_luminance(background_color)

        lighter = max(l1, l2)
        darker = min(l1, l2)

        contrast_ratio = (lighter + 0.05) / (darker + 0.05)

        # WCAG 2.1 Standards
        # Large text: >= 18pt or >= 14pt bold
        is_large_text = text_size >= 18.0

        # AA: 4.5:1 (normal), 3:1 (large)
        # AAA: 7:1 (normal), 4.5:1 (large)
        aa_threshold = 3.0 if is_large_text else 4.5
        aaa_threshold = 4.5 if is_large_text else 7.0

        meets_aa = contrast_ratio >= aa_threshold
        meets_aaa = contrast_ratio >= aaa_threshold

        return {
            "contrast_ratio": round(contrast_ratio, 2),
            "meets_aa": meets_aa,
            "meets_aaa": meets_aaa,
            "wcag_level": "AAA" if meets_aaa else ("AA" if meets_aa else "Fail"),
            "recommendation": self._get_contrast_recommendation(
                contrast_ratio, aa_threshold, aaa_threshold
            ),
            "foreground_rgb": foreground_color,
            "background_rgb": background_color
        }

    def _get_contrast_recommendation(
        self,
        ratio: float,
        aa_threshold: float,
        aaa_threshold: float
    ) -> str:
        """Get recommendation for contrast improvement"""
        if ratio >= aaa_threshold:
            return "Excellent contrast - exceeds WCAG AAA"
        elif ratio >= aa_threshold:
            return "Good contrast - meets WCAG AA"
        else:
            improvement_needed = aa_threshold - ratio
            return f"Insufficient contrast. Increase by {improvement_needed:.1f}x to meet WCAG AA"

    async def analyze_visual_complexity(
        self,
        content: str,
        has_images: int = 0,
        has_charts: int = 0,
        has_tables: int = 0,
        color_count: int = 0
    ) -> Dict[str, Any]:
        """
        Analyze visual complexity of content

        Args:
            content: Text content
            has_images: Number of images
            has_charts: Number of charts/graphs
            has_tables: Number of tables
            color_count: Number of distinct colors used

        Returns:
            Complexity analysis and recommendations
        """
        complexity_score = 0.0

        # Text complexity
        word_count = len(content.split())
        if word_count > 500:
            complexity_score += 0.3
        elif word_count > 200:
            complexity_score += 0.2
        else:
            complexity_score += 0.1

        # Visual elements
        visual_elements = has_images + has_charts + has_tables
        if visual_elements > 5:
            complexity_score += 0.3
        elif visual_elements > 2:
            complexity_score += 0.2
        else:
            complexity_score += 0.1

        # Color complexity
        if color_count > 10:
            complexity_score += 0.2
        elif color_count > 5:
            complexity_score += 0.15
        else:
            complexity_score += 0.1

        # Layout complexity (estimate from content structure)
        sections = len(re.findall(r'\n\s*\n', content))
        if sections > 10:
            complexity_score += 0.2
        else:
            complexity_score += 0.1

        complexity_level = (
            "low" if complexity_score < 0.4 else
            "medium" if complexity_score < 0.7 else
            "high"
        )

        recommendations = self._get_complexity_recommendations(
            complexity_level, visual_elements, color_count
        )

        return {
            "complexity_score": round(complexity_score, 2),
            "complexity_level": complexity_level,
            "word_count": word_count,
            "visual_elements": visual_elements,
            "color_count": color_count,
            "recommendations": recommendations
        }

    def _get_complexity_recommendations(
        self,
        level: str,
        visual_elements: int,
        color_count: int
    ) -> List[str]:
        """Get recommendations based on complexity"""
        recommendations = []

        if level == "high":
            recommendations.append("Consider breaking content into smaller sections")
            recommendations.append("Use progressive disclosure to reduce initial complexity")
            recommendations.append("Provide a simplified view option")

        if visual_elements > 5:
            recommendations.append("Add descriptive captions to all visual elements")
            recommendations.append("Consider providing text alternatives")

        if color_count > 10:
            recommendations.append("Reduce color palette for better color blindness support")
            recommendations.append("Use patterns or labels in addition to color coding")

        return recommendations

    async def optimize_for_screen_reader(
        self,
        content: str,
        has_headings: bool = False,
        has_landmarks: bool = False,
        has_alt_text: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze and optimize content for screen readers

        Args:
            content: Content to analyze
            has_headings: Whether content has semantic headings
            has_landmarks: Whether content has ARIA landmarks
            has_alt_text: Whether images have alt text

        Returns:
            Screen reader optimization analysis
        """
        issues = []
        recommendations = []

        # Check for semantic structure
        if not has_headings:
            issues.append("Missing semantic headings")
            recommendations.append("Add proper heading structure (h1, h2, h3)")

        if not has_landmarks:
            issues.append("Missing ARIA landmarks")
            recommendations.append("Add landmarks: main, navigation, complementary")

        if not has_alt_text:
            issues.append("Images missing alt text")
            recommendations.append("Provide descriptive alt text for all images")

        # Check for screen reader hostile patterns
        if re.search(r'click here|read more', content, re.IGNORECASE):
            issues.append("Non-descriptive link text detected")
            recommendations.append("Use descriptive link text instead of 'click here'")

        # Check for tables
        if 'table' in content.lower() or '|' in content:
            recommendations.append("Ensure tables have proper headers (th) and scope attributes")

        # Check for lists
        list_patterns = re.findall(r'^\s*[\-\*\d+\.]\s+', content, re.MULTILINE)
        if len(list_patterns) > 3:
            recommendations.append("Use semantic list elements (ul, ol) instead of plain text")

        accessibility_score = 1.0 - (len(issues) * 0.25)
        accessibility_score = max(0.0, min(1.0, accessibility_score))

        return {
            "accessibility_score": round(accessibility_score, 2),
            "issues": issues,
            "recommendations": recommendations,
            "screen_reader_friendly": len(issues) == 0,
            "improvement_areas": len(recommendations)
        }

    async def simulate_colorblindness(
        self,
        color_rgb: Tuple[int, int, int],
        colorblind_type: str = "deuteranopia"
    ) -> Dict[str, Any]:
        """
        Simulate how a color appears to colorblind users

        Args:
            color_rgb: RGB color tuple
            colorblind_type: Type of color blindness to simulate

        Returns:
            Simulated color and accessibility info
        """
        r, g, b = color_rgb

        # Simplified color blindness simulation matrices
        # In production, use proper transformation matrices
        if colorblind_type == "protanopia":
            # Red-blind: Remove red channel contribution
            simulated = (int(0.567 * r + 0.433 * g), g, b)
        elif colorblind_type == "deuteranopia":
            # Green-blind: Adjust green channel
            simulated = (r, int(0.625 * r + 0.375 * g), b)
        elif colorblind_type == "tritanopia":
            # Blue-blind: Adjust blue channel
            simulated = (r, g, int(0.95 * g + 0.05 * b))
        else:  # achromatopsia (grayscale)
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            simulated = (gray, gray, gray)

        return {
            "original_rgb": color_rgb,
            "simulated_rgb": simulated,
            "colorblind_type": colorblind_type,
            "type_description": self.colorblind_types.get(colorblind_type, "unknown"),
            "significant_change": self._color_distance(color_rgb, simulated) > 50,
            "recommendation": "Use text labels or patterns in addition to color"
        }

    def _color_distance(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> float:
        """Calculate Euclidean distance between two colors"""
        return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5
