"""
Basic usage example for the Neuroadaptive Accessibility Agent
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adk.agents.core import AccessibilityCoordinator
from adk.utils import SignalType, load_config


async def basic_example():
    """Basic usage example"""
    print("Neuroadaptive Accessibility Agent - Basic Example\n")

    # Initialize the coordinator
    coordinator = AccessibilityCoordinator()
    await coordinator.initialize()

    # Start a session
    user_id = "example_user_123"
    session_id = await coordinator.start_session(user_id)
    print(f"Started session: {session_id}\n")

    # Simulate user signals
    raw_signals = [
        (SignalType.EYE_TRACKING, 0.7, {"device": "webcam", "confidence": 0.9}),
        (SignalType.INTERACTION_TIMING, 0.65, {"avg_response_time_ms": 850}),
        (SignalType.MOUSE_MOVEMENT, 0.55, {"movement_pattern": "erratic"}),
    ]

    # Content to refine for accessibility
    content = """
    The neuroadaptive accessibility system provides real-time adaptations
    for users with diverse cognitive and sensory needs. It utilizes advanced
    machine learning algorithms to continuously monitor user state and adjust
    interface parameters accordingly.
    """

    # Process the interaction
    print("Processing user interaction...")
    result = await coordinator.process_user_interaction(
        raw_signals=raw_signals,
        user_id=user_id,
        content_to_refine=content,
        context={"page": "documentation", "section": "overview"}
    )

    # Display cognitive state
    print("\nCognitive State:")
    print(f"  Cognitive Load: {result['cognitive_state']['cognitive_load']:.2f}")
    print(f"  Attention Level: {result['cognitive_state']['attention_level']:.2f}")
    print(f"  Fatigue Index: {result['cognitive_state']['fatigue_index']:.2f}")
    print(f"  Confidence: {result['cognitive_state']['confidence']:.2f}")

    # Display UI adaptations
    print(f"\nUI Adaptations ({len(result['ui_adaptations'])} generated):")
    for adaptation in result['ui_adaptations'][:3]:  # Show top 3
        print(f"  - {adaptation['category']}: {adaptation['parameter']} = {adaptation['value']}")
        print(f"    Rationale: {adaptation['rationale']}")

    # Display content refinement
    if result.get('content_refinement'):
        refinement = result['content_refinement']
        print(f"\nContent Refinement:")
        print(f"  Quality Score: {refinement['final_quality_score']:.2f}")
        print(f"  Changes Made: {refinement['total_changes']}")

    # End session
    stats = await coordinator.end_session()
    print(f"\nSession ended. Total latency: {stats['statistics']['avg_latency_ms']:.2f}ms")

    # Cleanup
    await coordinator.close()


if __name__ == "__main__":
    asyncio.run(basic_example())
