"""
Neuroadaptive Accessibility Agent - Google ADK Implementation

This module implements a neuroadaptive accessibility system that adapts
to user cognitive and sensory needs in real-time using the Google Agent
Development Kit (ADK).

Architecture:
- Loop A: Signal normalization and preprocessing
- Loop B: State estimation with XGC-AVis integration
- Loop C: Content refinement (Factuality, Personalization, Coherence)
- Loop E: Logging and evaluation
- CMS: Continuum Memory System (mem0.ai)

Components:
- PerceptionPipeline: Processes user signals and environmental context
- AccessibilityPolicyLoop: Generates accessibility adaptations
- AccessibilityCoordinator: Orchestrates the complete system
"""

__version__ = "0.1.0"
__author__ = "DeepAgent Team"

from .agents.core.accessibility_coordinator import AccessibilityCoordinator
from .agents.core.perception_pipeline import PerceptionPipeline
from .agents.core.accessibility_policy_loop import AccessibilityPolicyLoop

__all__ = [
    "AccessibilityCoordinator",
    "PerceptionPipeline",
    "AccessibilityPolicyLoop",
]
