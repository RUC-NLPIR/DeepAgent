"""
Memory Store - mem0.ai Integration

This module provides integration with mem0.ai for persistent, contextual memory
storage for the neuroadaptive accessibility system.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from ...utils.schemas import MemoryRecord
from ...utils.config_loader import get_config_value
from ...utils.logger import get_logger


class MemoryStore:
    """
    Memory Store using mem0.ai

    Provides persistent storage for user preferences, accessibility history,
    interaction patterns, and cognitive profiles.

    Note: This is a wrapper around mem0.ai. In production, you would import
    and use the actual mem0ai package.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MemoryStore

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger("system")

        # Load configuration
        self.enabled = get_config_value("cms.enabled", True)
        self.memory_types = get_config_value(
            "cms.memory_types",
            ["user_preferences", "accessibility_history",
             "interaction_patterns", "cognitive_profiles"]
        )

        # In-memory storage (fallback when mem0 is not available)
        # In production, this would be replaced with actual mem0.ai client
        self._memory_cache: Dict[str, List[MemoryRecord]] = {
            mem_type: [] for mem_type in self.memory_types
        }

        # TODO: Initialize mem0.ai client when available
        # from mem0ai import MemoryClient
        # self.mem0_client = MemoryClient(api_key=get_config_value("cms.mem0_config.api_key"))

        self.logger.info("MemoryStore initialized (using in-memory fallback)")

    async def store_memory(
        self,
        memory_type: str,
        content: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        importance_score: float = 0.5,
        retention_days: Optional[int] = None
    ) -> MemoryRecord:
        """
        Store a memory record

        Args:
            memory_type: Type of memory (must be in configured memory_types)
            content: Memory content as dictionary
            user_id: Optional user identifier
            session_id: Optional session identifier
            importance_score: Importance score (0-1)
            retention_days: Days to retain memory, or None for default

        Returns:
            Created MemoryRecord
        """
        if memory_type not in self.memory_types:
            raise ValueError(f"Invalid memory type: {memory_type}")

        # Create memory record
        memory_id = f"{memory_type}_{datetime.now().timestamp()}"

        retention_until = None
        if retention_days is not None:
            retention_until = datetime.now() + timedelta(days=retention_days)

        memory_record = MemoryRecord(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            user_id=user_id,
            session_id=session_id,
            importance_score=importance_score,
            retention_until=retention_until
        )

        # Store in cache (in production, would store in mem0.ai)
        self._memory_cache[memory_type].append(memory_record)

        self.logger.debug(f"Stored memory: {memory_id} (type: {memory_type})")

        return memory_record

    async def retrieve_memories(
        self,
        memory_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[MemoryRecord]:
        """
        Retrieve memory records

        Args:
            memory_type: Filter by memory type, or None for all
            user_id: Filter by user ID
            session_id: Filter by session ID
            limit: Maximum number of records to return
            min_importance: Minimum importance score

        Returns:
            List of MemoryRecord objects
        """
        # Collect memories from cache
        if memory_type:
            memory_lists = [self._memory_cache.get(memory_type, [])]
        else:
            memory_lists = list(self._memory_cache.values())

        all_memories = []
        for mem_list in memory_lists:
            all_memories.extend(mem_list)

        # Filter
        filtered = []
        for memory in all_memories:
            # Check expiration
            if memory.retention_until and datetime.now() > memory.retention_until:
                continue

            # Check filters
            if user_id and memory.user_id != user_id:
                continue
            if session_id and memory.session_id != session_id:
                continue
            if memory.importance_score < min_importance:
                continue

            filtered.append(memory)

        # Sort by importance and recency
        filtered.sort(
            key=lambda m: (m.importance_score, m.created_at),
            reverse=True
        )

        return filtered[:limit]

    async def search_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[MemoryRecord]:
        """
        Search memories using semantic search

        Args:
            query: Search query
            memory_type: Filter by memory type
            user_id: Filter by user ID
            limit: Maximum results

        Returns:
            List of relevant MemoryRecord objects

        Note: In production, this would use mem0.ai's vector search.
        This implementation uses simple keyword matching as fallback.
        """
        memories = await self.retrieve_memories(
            memory_type=memory_type,
            user_id=user_id,
            limit=100  # Get more for searching
        )

        # Simple keyword matching (in production, use vector similarity)
        query_lower = query.lower()
        scored_memories = []

        for memory in memories:
            content_str = json.dumps(memory.content).lower()

            # Simple scoring based on keyword presence
            score = 0.0
            for word in query_lower.split():
                if word in content_str:
                    score += 1.0

            if score > 0:
                scored_memories.append((score, memory))

        # Sort by score
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        return [memory for _, memory in scored_memories[:limit]]

    async def update_memory(
        self,
        memory_id: str,
        content: Optional[Dict[str, Any]] = None,
        importance_score: Optional[float] = None
    ) -> Optional[MemoryRecord]:
        """
        Update an existing memory record

        Args:
            memory_id: Memory record ID
            content: New content, or None to keep existing
            importance_score: New importance score, or None to keep existing

        Returns:
            Updated MemoryRecord or None if not found
        """
        # Find memory in cache
        for mem_list in self._memory_cache.values():
            for i, memory in enumerate(mem_list):
                if memory.memory_id == memory_id:
                    # Update
                    if content is not None:
                        memory.content = content
                    if importance_score is not None:
                        memory.importance_score = importance_score
                    memory.updated_at = datetime.now()

                    self.logger.debug(f"Updated memory: {memory_id}")
                    return memory

        self.logger.warning(f"Memory not found for update: {memory_id}")
        return None

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory record

        Args:
            memory_id: Memory record ID

        Returns:
            True if deleted, False if not found
        """
        # Find and delete from cache
        for mem_list in self._memory_cache.values():
            for i, memory in enumerate(mem_list):
                if memory.memory_id == memory_id:
                    mem_list.pop(i)
                    self.logger.debug(f"Deleted memory: {memory_id}")
                    return True

        self.logger.warning(f"Memory not found for deletion: {memory_id}")
        return False

    async def cleanup_expired(self) -> int:
        """
        Clean up expired memory records

        Returns:
            Number of records deleted
        """
        deleted_count = 0
        now = datetime.now()

        for mem_type, mem_list in self._memory_cache.items():
            to_remove = []
            for i, memory in enumerate(mem_list):
                if memory.retention_until and now > memory.retention_until:
                    to_remove.append(i)

            # Remove in reverse order to maintain indices
            for i in reversed(to_remove):
                mem_list.pop(i)
                deleted_count += 1

        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} expired memories")

        return deleted_count

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory storage statistics

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_memories": sum(len(lst) for lst in self._memory_cache.values()),
            "by_type": {
                mem_type: len(mem_list)
                for mem_type, mem_list in self._memory_cache.items()
            },
            "storage_backend": "in_memory_fallback"  # Would be "mem0ai" in production
        }

        return stats
