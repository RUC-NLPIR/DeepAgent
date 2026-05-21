"""
Memory Manager - High-level Memory Management

Provides high-level memory management functionality for the neuroadaptive
accessibility system, including user profiles, preferences, and adaptation history.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...utils.schemas import (
    MemoryRecord,
    AccessibilityProfile,
    CognitiveState,
    AccessibilityAdaptation
)
from ...utils.config_loader import get_config_value
from ...utils.logger import get_logger
from .memory_store import MemoryStore


class MemoryManager:
    """
    High-level memory manager for CMS

    Manages user preferences, accessibility profiles, interaction patterns,
    and cognitive profiles using the underlying MemoryStore.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MemoryManager

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger("system")

        # Initialize memory store
        self.memory_store = MemoryStore(config)

        # Load retention policy
        self.short_term_hours = get_config_value("cms.retention_policy.short_term_hours", 24)
        self.long_term_days = get_config_value("cms.retention_policy.long_term_days", 90)
        self.aggregate_threshold = get_config_value("cms.retention_policy.aggregate_threshold", 10)

        self.logger.info("MemoryManager initialized")

    async def save_user_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: Any,
        importance: float = 0.7
    ) -> MemoryRecord:
        """
        Save a user preference

        Args:
            user_id: User identifier
            preference_key: Preference key
            preference_value: Preference value
            importance: Importance score

        Returns:
            Created MemoryRecord
        """
        content = {
            "preference_key": preference_key,
            "preference_value": preference_value,
            "timestamp": datetime.now().isoformat()
        }

        return await self.memory_store.store_memory(
            memory_type="user_preferences",
            content=content,
            user_id=user_id,
            importance_score=importance,
            retention_days=self.long_term_days
        )

    async def get_user_preferences(
        self,
        user_id: str,
        preference_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get user preferences

        Args:
            user_id: User identifier
            preference_key: Specific preference key, or None for all

        Returns:
            Dictionary of preferences
        """
        memories = await self.memory_store.retrieve_memories(
            memory_type="user_preferences",
            user_id=user_id,
            limit=100
        )

        preferences = {}
        for memory in memories:
            key = memory.content.get("preference_key")
            value = memory.content.get("preference_value")

            if preference_key is None or key == preference_key:
                # Use most recent value for each key
                if key not in preferences:
                    preferences[key] = value

        return preferences

    async def save_accessibility_profile(
        self,
        profile: AccessibilityProfile
    ) -> MemoryRecord:
        """
        Save an accessibility profile

        Args:
            profile: AccessibilityProfile object

        Returns:
            Created MemoryRecord
        """
        content = {
            "profile_id": profile.profile_id,
            "profile_name": profile.profile_name,
            "settings": profile.settings,
            "cognitive_preferences": profile.cognitive_preferences,
            "sensory_preferences": profile.sensory_preferences,
            "interaction_preferences": profile.interaction_preferences,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat()
        }

        return await self.memory_store.store_memory(
            memory_type="user_preferences",
            content=content,
            user_id=profile.user_id,
            importance_score=1.0,  # Profiles are highly important
            retention_days=self.long_term_days
        )

    async def get_accessibility_profile(
        self,
        user_id: str,
        profile_id: Optional[str] = None
    ) -> Optional[AccessibilityProfile]:
        """
        Get accessibility profile for a user

        Args:
            user_id: User identifier
            profile_id: Specific profile ID, or None for default

        Returns:
            AccessibilityProfile or None
        """
        memories = await self.memory_store.retrieve_memories(
            memory_type="user_preferences",
            user_id=user_id,
            limit=50
        )

        for memory in memories:
            content = memory.content
            if "profile_id" in content:
                if profile_id is None or content["profile_id"] == profile_id:
                    return AccessibilityProfile(
                        profile_id=content["profile_id"],
                        profile_name=content["profile_name"],
                        user_id=user_id,
                        settings=content.get("settings", {}),
                        cognitive_preferences=content.get("cognitive_preferences"),
                        sensory_preferences=content.get("sensory_preferences"),
                        interaction_preferences=content.get("interaction_preferences"),
                        created_at=datetime.fromisoformat(content["created_at"]),
                        updated_at=datetime.fromisoformat(content["updated_at"])
                    )

        return None

    async def save_adaptation_history(
        self,
        user_id: str,
        session_id: str,
        adaptation: AccessibilityAdaptation,
        cognitive_state: CognitiveState
    ) -> MemoryRecord:
        """
        Save adaptation history

        Args:
            user_id: User identifier
            session_id: Session identifier
            adaptation: AccessibilityAdaptation applied
            cognitive_state: CognitiveState at time of adaptation

        Returns:
            Created MemoryRecord
        """
        content = {
            "adaptation_id": adaptation.adaptation_id,
            "category": adaptation.category,
            "parameter": adaptation.parameter,
            "value": adaptation.value,
            "confidence": adaptation.confidence,
            "rationale": adaptation.rationale,
            "cognitive_state": {
                "cognitive_load": cognitive_state.cognitive_load,
                "attention_level": cognitive_state.attention_level,
                "fatigue_index": cognitive_state.fatigue_index,
                "stress_level": cognitive_state.stress_level,
                "reading_comprehension": cognitive_state.reading_comprehension
            },
            "timestamp": adaptation.timestamp.isoformat()
        }

        return await self.memory_store.store_memory(
            memory_type="accessibility_history",
            content=content,
            user_id=user_id,
            session_id=session_id,
            importance_score=adaptation.confidence,
            retention_days=self.long_term_days
        )

    async def get_adaptation_history(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get adaptation history

        Args:
            user_id: User identifier
            session_id: Optional session identifier
            limit: Maximum records to return

        Returns:
            List of adaptation records
        """
        memories = await self.memory_store.retrieve_memories(
            memory_type="accessibility_history",
            user_id=user_id,
            session_id=session_id,
            limit=limit
        )

        return [memory.content for memory in memories]

    async def save_interaction_pattern(
        self,
        user_id: str,
        session_id: str,
        pattern_type: str,
        pattern_data: Dict[str, Any],
        importance: float = 0.5
    ) -> MemoryRecord:
        """
        Save interaction pattern

        Args:
            user_id: User identifier
            session_id: Session identifier
            pattern_type: Type of pattern
            pattern_data: Pattern data
            importance: Importance score

        Returns:
            Created MemoryRecord
        """
        content = {
            "pattern_type": pattern_type,
            "pattern_data": pattern_data,
            "timestamp": datetime.now().isoformat()
        }

        return await self.memory_store.store_memory(
            memory_type="interaction_patterns",
            content=content,
            user_id=user_id,
            session_id=session_id,
            importance_score=importance,
            retention_days=self.short_term_hours / 24  # Convert hours to days
        )

    async def get_interaction_patterns(
        self,
        user_id: str,
        pattern_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get interaction patterns

        Args:
            user_id: User identifier
            pattern_type: Filter by pattern type
            limit: Maximum records

        Returns:
            List of pattern records
        """
        memories = await self.memory_store.retrieve_memories(
            memory_type="interaction_patterns",
            user_id=user_id,
            limit=limit
        )

        patterns = []
        for memory in memories:
            if pattern_type is None or memory.content.get("pattern_type") == pattern_type:
                patterns.append(memory.content)

        return patterns

    async def save_cognitive_profile(
        self,
        user_id: str,
        cognitive_state: CognitiveState,
        session_id: Optional[str] = None
    ) -> MemoryRecord:
        """
        Save cognitive profile snapshot

        Args:
            user_id: User identifier
            cognitive_state: Current cognitive state
            session_id: Optional session identifier

        Returns:
            Created MemoryRecord
        """
        content = {
            "cognitive_load": cognitive_state.cognitive_load,
            "attention_level": cognitive_state.attention_level,
            "fatigue_index": cognitive_state.fatigue_index,
            "stress_level": cognitive_state.stress_level,
            "reading_comprehension": cognitive_state.reading_comprehension,
            "confidence": cognitive_state.confidence,
            "timestamp": cognitive_state.timestamp.isoformat()
        }

        return await self.memory_store.store_memory(
            memory_type="cognitive_profiles",
            content=content,
            user_id=user_id,
            session_id=session_id,
            importance_score=cognitive_state.confidence,
            retention_days=self.long_term_days
        )

    async def get_cognitive_profile_average(
        self,
        user_id: str,
        limit: int = 100
    ) -> Optional[CognitiveState]:
        """
        Get average cognitive profile for a user

        Args:
            user_id: User identifier
            limit: Number of recent states to average

        Returns:
            Average CognitiveState or None
        """
        memories = await self.memory_store.retrieve_memories(
            memory_type="cognitive_profiles",
            user_id=user_id,
            limit=limit
        )

        if not memories:
            return None

        import numpy as np

        avg_state = CognitiveState(
            cognitive_load=float(np.mean([m.content["cognitive_load"] for m in memories])),
            attention_level=float(np.mean([m.content["attention_level"] for m in memories])),
            fatigue_index=float(np.mean([m.content["fatigue_index"] for m in memories])),
            stress_level=float(np.mean([m.content["stress_level"] for m in memories])),
            reading_comprehension=float(np.mean([m.content["reading_comprehension"] for m in memories])),
            confidence=float(np.mean([m.content["confidence"] for m in memories]))
        )

        return avg_state

    async def search_relevant_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5
    ) -> List[MemoryRecord]:
        """
        Search for relevant memories across all types

        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum results

        Returns:
            List of relevant MemoryRecord objects
        """
        return await self.memory_store.search_memories(
            query=query,
            user_id=user_id,
            limit=limit
        )

    async def cleanup(self):
        """Clean up expired memories"""
        deleted = await self.memory_store.cleanup_expired()
        self.logger.info(f"Memory cleanup completed. Removed {deleted} records.")
        return deleted

    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return await self.memory_store.get_statistics()
