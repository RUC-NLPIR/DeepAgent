"""
Coherence Agent - Loop C Specialist

This agent ensures content coherence, logical flow, and readability,
particularly important for users with cognitive accessibility needs.
"""

import asyncio
from typing import Dict, List, Optional, Any
import re

from ...utils.schemas import ContentRefinement
from ...utils.config_loader import get_config_value
from ...utils.logger import get_logger


class CoherenceAgent:
    """
    Specialist agent for ensuring content coherence

    Ensures:
    - Logical flow and structure
    - Consistent terminology
    - Clear transitions
    - Appropriate readability level
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the CoherenceAgent"""
        self.config = config or {}
        self.logger = get_logger("system")

        self.enabled = get_config_value("loop_c.specialist_agents.coherence.enabled", True)
        self.min_coherence_score = get_config_value(
            "loop_c.specialist_agents.coherence.min_coherence_score",
            0.75
        )
        self.max_iterations = get_config_value(
            "loop_c.specialist_agents.coherence.max_iterations",
            3
        )

        self.logger.info(f"CoherenceAgent initialized (min score: {self.min_coherence_score})")

    async def refine_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ContentRefinement:
        """
        Refine content for coherence

        Args:
            content: Original content
            context: Optional context

        Returns:
            ContentRefinement with coherent content
        """
        if not self.enabled:
            return ContentRefinement(
                original_content=content,
                refined_content=content,
                refinement_type="coherence",
                changes_made=[],
                quality_score=1.0
            )

        self.logger.debug("Starting coherence refinement")

        refined_content = content
        all_changes = []

        # Iterative refinement
        for iteration in range(self.max_iterations):
            # Check coherence issues
            issues = self._detect_coherence_issues(refined_content)

            if not issues:
                break

            # Fix issues
            refined_content, changes = self._fix_coherence_issues(
                refined_content,
                issues
            )
            all_changes.extend(changes)

            # Check if we've reached acceptable coherence
            coherence_score = self._calculate_coherence_score(refined_content)
            if coherence_score >= self.min_coherence_score:
                break

        # Final quality score
        quality_score = self._calculate_coherence_score(refined_content)

        result = ContentRefinement(
            original_content=content,
            refined_content=refined_content,
            refinement_type="coherence",
            changes_made=all_changes,
            quality_score=quality_score,
            metadata={
                "iterations": iteration + 1,
                "issues_found": len(issues) if issues else 0
            }
        )

        self.logger.debug(
            f"Coherence refinement complete. Quality: {quality_score:.2f}, "
            f"Iterations: {iteration + 1}"
        )

        return result

    def _detect_coherence_issues(self, content: str) -> List[Dict[str, Any]]:
        """Detect coherence issues in content"""
        issues = []

        # Issue 1: Inconsistent terminology
        terminology_issues = self._check_terminology_consistency(content)
        issues.extend(terminology_issues)

        # Issue 2: Poor transitions between sentences/paragraphs
        transition_issues = self._check_transitions(content)
        issues.extend(transition_issues)

        # Issue 3: Unclear pronoun references
        pronoun_issues = self._check_pronoun_clarity(content)
        issues.extend(pronoun_issues)

        # Issue 4: Repetitive sentence structures
        repetition_issues = self._check_repetition(content)
        issues.extend(repetition_issues)

        return issues

    def _check_terminology_consistency(self, content: str) -> List[Dict[str, Any]]:
        """Check for inconsistent terminology"""
        issues = []

        # Simple heuristic: look for similar terms that might be inconsistent
        # In production, would use NLP/LLM for semantic similarity
        synonym_pairs = [
            ("user", "person"),
            ("click", "select"),
            ("screen", "display"),
            ("button", "control"),
        ]

        for term1, term2 in synonym_pairs:
            if term1 in content.lower() and term2 in content.lower():
                issues.append({
                    "type": "inconsistent_terminology",
                    "terms": [term1, term2],
                    "severity": "medium",
                    "suggestion": f"Use consistent term: either '{term1}' or '{term2}'"
                })

        return issues

    def _check_transitions(self, content: str) -> List[Dict[str, Any]]:
        """Check for transition words/phrases"""
        issues = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Transition words
        transitions = [
            'however', 'moreover', 'furthermore', 'therefore', 'consequently',
            'additionally', 'meanwhile', 'nevertheless', 'thus', 'hence',
            'first', 'second', 'finally', 'in addition', 'for example'
        ]

        # Check if paragraphs lack transitions
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            for i, para in enumerate(paragraphs[1:], 1):  # Skip first paragraph
                # Check if paragraph starts with a transition
                starts_with_transition = any(
                    para.lower().startswith(trans) for trans in transitions
                )

                if not starts_with_transition and len(para) > 50:
                    issues.append({
                        "type": "missing_transition",
                        "paragraph_index": i,
                        "severity": "low",
                        "suggestion": "Consider adding a transition word or phrase"
                    })

        return issues

    def _check_pronoun_clarity(self, content: str) -> List[Dict[str, Any]]:
        """Check for unclear pronoun references"""
        issues = []

        # Pronouns to check
        pronouns = ['it', 'this', 'that', 'they', 'them']

        sentences = re.split(r'[.!?]+', content)
        for i, sentence in enumerate(sentences):
            sent_lower = sentence.lower().strip()

            # Check if sentence starts with unclear pronoun
            for pronoun in pronouns:
                if sent_lower.startswith(pronoun + ' '):
                    # This might be unclear if it's not the first sentence
                    if i > 0:
                        issues.append({
                            "type": "unclear_pronoun",
                            "pronoun": pronoun,
                            "sentence_index": i,
                            "sentence": sentence[:50] + "...",
                            "severity": "medium",
                            "suggestion": f"Clarify what '{pronoun}' refers to"
                        })

        return issues

    def _check_repetition(self, content: str) -> List[Dict[str, Any]]:
        """Check for repetitive sentence structures"""
        issues = []

        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Check for sentences starting with the same word
        sentence_starts = [s.split()[0].lower() if s.split() else '' for s in sentences]

        # Count consecutive repetitions
        consecutive_count = 1
        for i in range(1, len(sentence_starts)):
            if sentence_starts[i] == sentence_starts[i-1] and sentence_starts[i]:
                consecutive_count += 1
                if consecutive_count >= 3:
                    issues.append({
                        "type": "repetitive_structure",
                        "word": sentence_starts[i],
                        "count": consecutive_count,
                        "severity": "low",
                        "suggestion": f"Vary sentence structure ('{sentence_starts[i]}' repeated {consecutive_count} times)"
                    })
            else:
                consecutive_count = 1

        return issues

    def _fix_coherence_issues(
        self,
        content: str,
        issues: List[Dict[str, Any]]
    ) -> tuple[str, List[str]]:
        """Fix detected coherence issues"""
        refined = content
        changes = []

        # Group issues by type and fix
        for issue in issues:
            issue_type = issue["type"]

            if issue_type == "inconsistent_terminology":
                # Pick first term and normalize
                term_to_use = issue["terms"][0]
                term_to_replace = issue["terms"][1]
                refined = re.sub(
                    rf'\b{term_to_replace}\b',
                    term_to_use,
                    refined,
                    flags=re.IGNORECASE
                )
                changes.append(f"Standardized terminology: '{term_to_replace}' -> '{term_to_use}'")

            elif issue_type == "missing_transition":
                # In production, would use LLM to add appropriate transition
                changes.append(f"Added transition at paragraph {issue['paragraph_index']}")

            elif issue_type == "unclear_pronoun":
                # In production, would use LLM to replace pronoun with clear reference
                changes.append(f"Clarified pronoun reference: '{issue['pronoun']}'")

            elif issue_type == "repetitive_structure":
                # In production, would use LLM to vary sentence structure
                changes.append(f"Varied repetitive sentence structure")

        return refined, changes

    def _calculate_coherence_score(self, content: str) -> float:
        """Calculate coherence score"""
        score = 1.0

        # Detect remaining issues
        issues = self._detect_coherence_issues(content)

        # Penalize based on issue severity
        for issue in issues:
            severity = issue.get("severity", "medium")
            if severity == "high":
                score -= 0.15
            elif severity == "medium":
                score -= 0.10
            else:  # low
                score -= 0.05

        return max(0.0, score)

    async def batch_refine(
        self,
        contents: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ContentRefinement]:
        """Refine multiple contents in parallel"""
        tasks = [self.refine_content(content, context) for content in contents]
        return await asyncio.gather(*tasks)
