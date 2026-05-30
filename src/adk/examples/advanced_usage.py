"""
Advanced usage example showing custom profiles and memory integration
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adk.agents.core import AccessibilityCoordinator
from adk.utils import SignalType, AccessibilityProfile
from adk.tools.memory import MemoryManager


async def advanced_example():
    """Advanced example with custom profiles and memory"""
    print("Neuroadaptive Accessibility Agent - Advanced Example\n")

    # Initialize components
    coordinator = AccessibilityCoordinator()
    await coordinator.initialize()

    memory_manager = MemoryManager()

    user_id = "advanced_user_456"

    # Create custom accessibility profile
    profile = AccessibilityProfile(
        profile_id="custom_profile_1",
        profile_name="High Cognitive Support",
        user_id=user_id,
        settings={
            "text_size": 1.3,
            "contrast": "high",
            "color_scheme": "dark",
            "simplified_language": True,
            "max_sentence_length": 15,
            "layout_density": "sparse"
        },
        cognitive_preferences={
            "prefer_visual_aids": True,
            "prefer_audio_descriptions": False,
            "reading_level": "elementary"
        }
    )

    # Save profile to memory
    await memory_manager.save_accessibility_profile(profile)
    print(f"Created accessibility profile: {profile.profile_name}\n")

    # Save user preference
    await memory_manager.save_user_preference(
        user_id,
        "preferred_font",
        "OpenDyslexic",
        importance=0.9
    )

    # Start session
    await coordinator.start_session(user_id)

    # Process multiple interactions
    for i in range(3):
        print(f"\nInteraction {i+1}:")

        # Varying signals to simulate changing state
        import random
        raw_signals = [
            (SignalType.EYE_TRACKING, random.uniform(0.4, 0.8), {}),
            (SignalType.INTERACTION_TIMING, random.uniform(0.5, 0.9), {}),
            (SignalType.SPEECH_PATTERNS, random.uniform(0.3, 0.7), {}),
        ]

        result = await coordinator.process_user_interaction(
            raw_signals,
            user_id,
            content_to_refine="Sample content for adaptation testing.",
            context={"interaction_number": i+1}
        )

        print(f"  Cognitive Load: {result['cognitive_state']['cognitive_load']:.2f}")
        print(f"  Adaptations Applied: {len(result['ui_adaptations'])}")

        await asyncio.sleep(0.5)

    # Get adaptation history
    history = await memory_manager.get_adaptation_history(user_id, limit=10)
    print(f"\nAdaptation History: {len(history)} records")

    # Get cognitive profile average
    avg_profile = await memory_manager.get_cognitive_profile_average(user_id)
    if avg_profile:
        print(f"\nAverage Cognitive Profile:")
        print(f"  Avg Cognitive Load: {avg_profile.cognitive_load:.2f}")
        print(f"  Avg Attention: {avg_profile.attention_level:.2f}")

    # Search memory
    relevant_memories = await memory_manager.search_relevant_memories(
        "high cognitive load",
        user_id
    )
    print(f"\nRelevant Memories Found: {len(relevant_memories)}")

    # End session and cleanup
    await coordinator.end_session()
    await coordinator.close()
    print("\nAdvanced example complete!")


if __name__ == "__main__":
    asyncio.run(advanced_example())
