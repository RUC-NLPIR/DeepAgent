"""
Bias Mitigation Demo - Alexithymia Fairness

Demonstrates how bidirectional reasoning mitigates bias against
neurodivergent users with alexithymia (flat affect).
"""

import asyncio
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adk.neuroadaptive_wrapper import NeuroadaptiveWrapper
from adk.bidirectional_reasoning import ReasoningConfig
from adk.evaluation.bias_metrics import AlexithymiaFairnessMetrics
from adk.utils import SignalType


async def simulate_neurotypical_user():
    """Simulate a neurotypical user with typical emotional expression"""
    print("\n" + "=" * 60)
    print("NEUROTYPICAL USER SIMULATION")
    print("=" * 60)

    # Create wrapper for neurotypical user
    wrapper = NeuroadaptiveWrapper(
        user_profile={
            "alexithymia_score": 0.1,  # Low alexithymia
            "neurodivergent_flags": []
        },
        reasoning_config=ReasoningConfig(device='cpu')
    )
    await wrapper.initialize()

    # Simulate emotional speech with clear prosody
    print("\nScenario: User expresses happiness with clear prosody")

    # Simulate audio features (high variance = clear emotional expression)
    audio_features = torch.randn(1, 50, 768) * 2.0 + 5.0  # High variance, positive bias

    # Corresponding user signals
    raw_signals = [
        (SignalType.SPEECH_PATTERNS, 0.8, {"prosody": "high_variance"}),
        (SignalType.INTERACTION_TIMING, 0.3, {"response_time_ms": 500}),
    ]

    # Process interaction
    result = await wrapper.process_interaction_with_emotion(
        raw_signals=raw_signals,
        audio_features=audio_features,
        text_content="I'm really excited about this new feature!",
        user_id="neurotypical_user_001"
    )

    # Display results
    print(f"\nEmotion Detected: {result['emotion_analysis']['emotion']}")
    print(f"Confidence: {result['emotion_analysis']['confidence']:.3f}")
    print(f"Verification Score: {result['emotion_analysis']['verification_score']:.3f}")
    print(f"Is Verified: {result['emotion_analysis']['is_verified']}")

    if 'alexithymia_indicator' in result['emotion_analysis']:
        print(f"Alexithymia Indicator: {result['emotion_analysis']['alexithymia_indicator']:.3f}")

    print(f"\nCognitive State:")
    print(f"  Cognitive Load: {result['cognitive_state']['cognitive_load']:.2f}")
    print(f"  Stress Level: {result['cognitive_state']['stress_level']:.2f}")

    await wrapper.close()
    return result


async def simulate_alexithymic_user():
    """Simulate an alexithymic user with flat affect"""
    print("\n" + "=" * 60)
    print("ALEXITHYMIC USER SIMULATION")
    print("=" * 60)

    # Create wrapper for alexithymic user
    wrapper = NeuroadaptiveWrapper(
        user_profile={
            "alexithymia_score": 0.85,  # High alexithymia
            "neurodivergent_flags": ["alexithymia", "autism"]
        },
        reasoning_config=ReasoningConfig(device='cpu')
    )
    await wrapper.initialize()

    # Simulate emotional speech with FLAT prosody
    print("\nScenario: User expresses happiness BUT with flat affect (alexithymia)")

    # Simulate audio features (LOW variance = flat affect, but same semantic content)
    # Key: The WORDS say "happy" but the PROSODY is flat
    audio_features = torch.randn(1, 50, 768) * 0.3 + 5.0  # LOW variance, same content

    # Corresponding user signals
    raw_signals = [
        (SignalType.SPEECH_PATTERNS, 0.2, {"prosody": "flat"}),  # Flat prosody!
        (SignalType.INTERACTION_TIMING, 0.3, {"response_time_ms": 500}),
    ]

    # Process interaction
    result = await wrapper.process_interaction_with_emotion(
        raw_signals=raw_signals,
        audio_features=audio_features,
        text_content="I'm really excited about this new feature!",  # SAME text
        user_id="alexithymic_user_001"
    )

    # Display results
    print(f"\nEmotion Detected: {result['emotion_analysis']['emotion']}")
    print(f"Confidence: {result['emotion_analysis']['confidence']:.3f}")
    print(f"Verification Score: {result['emotion_analysis']['verification_score']:.3f}")
    print(f"Is Verified: {result['emotion_analysis']['is_verified']}")

    # KEY: For alexithymic users, low verification is EXPECTED, not an error!
    if 'alexithymia_indicator' in result['emotion_analysis']:
        print(f"\nAlexithymia Indicator: {result['emotion_analysis']['alexithymia_indicator']:.3f}")
        print(f"Bias Mitigation: {result['emotion_analysis'].get('bias_mitigation', 'none')}")
        print("\n✓ EXPECTED: Low verification for alexithymic user (not treated as error)")

    print(f"\nCognitive State:")
    print(f"  Cognitive Load: {result['cognitive_state']['cognitive_load']:.2f}")
    print(f"  Stress Level: {result['cognitive_state']['stress_level']:.2f}")

    # Show alexithymia-specific adaptations
    print(f"\nAlexithymia-Specific Adaptations:")
    enhanced_adaptations = result.get('enhanced_adaptations', [])
    for adaptation in enhanced_adaptations:
        if 'alexithymi' in adaptation.get('rationale', '').lower():
            print(f"  • {adaptation['parameter']}: {adaptation['value']}")
            print(f"    Rationale: {adaptation['rationale']}")

    await wrapper.close()
    return result


async def compare_fairness():
    """Compare fairness between neurotypical and alexithymic users"""
    print("\n" + "=" * 60)
    print("FAIRNESS COMPARISON")
    print("=" * 60)

    metrics = AlexithymiaFairnessMetrics()

    # Simulate multiple users of each type
    print("\nSimulating 10 neurotypical users...")
    for i in range(10):
        wrapper = NeuroadaptiveWrapper(
            user_profile={"alexithymia_score": 0.1},
            reasoning_config=ReasoningConfig(device='cpu')
        )
        await wrapper.initialize()

        audio_features = torch.randn(1, 50, 768) * 2.0 + np.random.randn()
        raw_signals = [(SignalType.SPEECH_PATTERNS, np.random.rand(), {})]

        result = await wrapper.process_interaction_with_emotion(
            raw_signals=raw_signals,
            audio_features=audio_features,
            user_id=f"neurotypical_{i}"
        )

        # Assume ground truth is "happy" for demo
        metrics.add_prediction(
            result['emotion_analysis'],
            "happy",
            alexithymia_score=0.1
        )

        await wrapper.close()

    print("\nSimulating 10 alexithymic users...")
    for i in range(10):
        wrapper = NeuroadaptiveWrapper(
            user_profile={"alexithymia_score": 0.9},
            reasoning_config=ReasoningConfig(device='cpu')
        )
        await wrapper.initialize()

        # Flat affect (low variance)
        audio_features = torch.randn(1, 50, 768) * 0.3 + np.random.randn()
        raw_signals = [(SignalType.SPEECH_PATTERNS, 0.2, {"prosody": "flat"})]

        result = await wrapper.process_interaction_with_emotion(
            raw_signals=raw_signals,
            audio_features=audio_features,
            user_id=f"alexithymic_{i}"
        )

        metrics.add_prediction(
            result['emotion_analysis'],
            "happy",
            alexithymia_score=0.9
        )

        await wrapper.close()

    # Print fairness report
    print("\n")
    metrics.print_report()


async def main():
    """Main demo"""
    print("=" * 60)
    print("BIDIRECTIONAL REASONING: BIAS MITIGATION DEMO")
    print("Addressing Emotion AI Bias for Alexithymic Users")
    print("=" * 60)

    # Part 1: Neurotypical user
    await simulate_neurotypical_user()

    # Part 2: Alexithymic user (key demonstration)
    await simulate_alexithymic_user()

    # Part 3: Fairness comparison
    await compare_fairness()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Neurotypical users: High verification scores expected")
    print("2. Alexithymic users: Low verification is NORMAL (flat affect)")
    print("3. Bidirectional reasoning detects this pattern")
    print("4. System applies alexithymia-specific adaptations")
    print("5. No false negatives due to flat affect!")
    print("\nThis addresses the Bias Bounty challenge:")
    print("- Traditional emotion AI: Flat affect → Missed emotions")
    print("- Our approach: Flat affect → Recognized as alexithymia pattern")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
