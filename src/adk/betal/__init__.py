"""
BeTaL: Benchmark Tailoring via LLM Feedback for Accessibility Fairness

Based on Dsouza et al. (arXiv:2510.25039v1)

Automated benchmark design for testing emotion AI fairness across neurotypes.
"""

from .accessibility_betal import AccessibilityBeTaL, BeTaLConfig
from .betal_comparison import BeTaLComparison, compare_to_baselines

__all__ = [
    "AccessibilityBeTaL",
    "BeTaLConfig",
    "BeTaLComparison",
    "compare_to_baselines"
]
