#!/usr/bin/env python3
"""
Entry point for running the Neuroadaptive Accessibility Agent

Usage:
    python run_accessibility_agent.py [--config CONFIG_PATH] [--user-id USER_ID]

Example:
    python run_accessibility_agent.py --user-id user123
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adk.agents.core import AccessibilityCoordinator
from adk.utils import load_config, setup_logging, SignalType
from adk.utils.logger import get_logger


async def demo_single_interaction(coordinator: AccessibilityCoordinator, user_id: str):
    """
    Demonstrate a single user interaction

    Args:
        coordinator: AccessibilityCoordinator instance
        user_id: User identifier
    """
    logger = get_logger("system")

    # Simulate raw user signals
    raw_signals = [
        (SignalType.EYE_TRACKING, 0.65, {"device": "webcam"}),
        (SignalType.INTERACTION_TIMING, 0.72, {"last_click_delay_ms": 1200}),
        (SignalType.MOUSE_MOVEMENT, 0.58, {"movement_speed": "slow"}),
    ]

    # Sample content to refine
    content = (
        "The implementation of the neuroadaptive accessibility system utilizes "
        "sophisticated algorithms to facilitate the optimization of user interfaces "
        "based on cognitive load metrics."
    )

    # Process interaction
    result = await coordinator.process_user_interaction(
        raw_signals=raw_signals,
        user_id=user_id,
        content_to_refine=content,
        context={"page": "dashboard", "task": "reading_documentation"}
    )

    # Display results
    logger.info("=" * 60)
    logger.info("ACCESSIBILITY ADAPTATION RESULT")
    logger.info("=" * 60)

    # Cognitive State
    logger.info("\nCognitive State:")
    for key, value in result["cognitive_state"].items():
        logger.info(f"  {key}: {value:.3f}")

    # UI Adaptations
    logger.info(f"\nUI Adaptations ({len(result['ui_adaptations'])} total):")
    for adaptation in result["ui_adaptations"]:
        logger.info(
            f"  [{adaptation['priority']}] {adaptation['category']}/{adaptation['parameter']}: "
            f"{adaptation['value']}"
        )
        logger.info(f"      Rationale: {adaptation['rationale']}")

    # Content Refinement
    if result.get("content_refinement"):
        refinement = result["content_refinement"]
        logger.info(f"\nContent Refinement:")
        logger.info(f"  Iterations: {refinement['iterations_completed']}")
        logger.info(f"  Final Score: {refinement['final_quality_score']:.3f}")
        logger.info(f"  Total Changes: {refinement['total_changes']}")
        logger.info(f"\nOriginal:")
        logger.info(f"  {refinement['original_content'][:100]}...")
        logger.info(f"\nRefined:")
        logger.info(f"  {refinement['refined_content'][:100]}...")

    # Metrics
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Latency: {result['metrics']['latency_ms']:.2f}ms")
    logger.info(f"  Accessibility Score: {result['metrics']['accessibility_score']:.3f}")

    logger.info("=" * 60)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Neuroadaptive Accessibility Agent"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="demo_user",
        help="User identifier"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "interactive", "stream"],
        default="demo",
        help="Run mode"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        load_config(args.config)

    # Setup logging
    setup_logging()
    logger = get_logger("system")

    logger.info("Starting Neuroadaptive Accessibility Agent")
    logger.info(f"User ID: {args.user_id}")
    logger.info(f"Mode: {args.mode}")

    # Initialize coordinator
    coordinator = AccessibilityCoordinator()
    await coordinator.initialize()

    try:
        if args.mode == "demo":
            # Run single interaction demo
            await demo_single_interaction(coordinator, args.user_id)

        elif args.mode == "interactive":
            # Interactive mode
            logger.info("\nInteractive mode - type 'quit' to exit")
            await coordinator.start_session(args.user_id)

            while True:
                user_input = input("\nPress Enter to process interaction (or 'quit'): ")
                if user_input.lower() == 'quit':
                    break

                # Simulate signals (in production, would come from sensors)
                import random
                raw_signals = [
                    (SignalType.EYE_TRACKING, random.uniform(0.3, 0.9), {}),
                    (SignalType.INTERACTION_TIMING, random.uniform(0.4, 0.8), {}),
                ]

                result = await coordinator.process_user_interaction(
                    raw_signals,
                    args.user_id
                )

                logger.info(f"Cognitive Load: {result['cognitive_state']['cognitive_load']:.3f}")
                logger.info(f"Adaptations: {len(result['ui_adaptations'])}")

        elif args.mode == "stream":
            # Streaming mode (for continuous processing)
            logger.info("\nStreaming mode - processing for 30 seconds")

            # Create signal stream
            signal_stream = asyncio.Queue()

            # Producer task to simulate signals
            async def signal_producer():
                import random
                for _ in range(30):  # 30 iterations
                    signals = [
                        (SignalType.EYE_TRACKING, random.uniform(0.3, 0.9), {}),
                        (SignalType.INTERACTION_TIMING, random.uniform(0.4, 0.8), {}),
                    ]
                    await signal_stream.put(signals)
                    await asyncio.sleep(1.0)

            # Run adaptive loop
            producer_task = asyncio.create_task(signal_producer())
            await coordinator.run_adaptive_loop(
                args.user_id,
                signal_stream,
                max_duration_seconds=35
            )
            await producer_task

        # End session and show statistics
        stats = await coordinator.end_session()
        if stats:
            logger.info("\nSession Statistics:")
            for key, value in stats.get("statistics", {}).items():
                logger.info(f"  {key}: {value}")

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await coordinator.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
