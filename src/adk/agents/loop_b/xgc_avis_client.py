"""
XGC-AVis Integration Client

Client for interfacing with the XGC-AVis (eXtended Generalized Cognitive -
Adaptive Visualization) service for advanced cognitive state estimation.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...utils.schemas import UserSignal, CognitiveState
from ...utils.config_loader import get_config_value
from ...utils.logger import get_logger


class XGCAVisClient:
    """
    Client for XGC-AVis cognitive state estimation service

    This client communicates with an external XGC-AVis service to estimate
    user cognitive states based on multimodal signals.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize XGC-AVis client

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger("system")

        # Load configuration
        self.endpoint = get_config_value(
            "loop_b.xgc_avis.endpoint",
            "http://localhost:8080/xgc-avis"
        )
        self.timeout = get_config_value("loop_b.xgc_avis.timeout", 5.0)
        self.retry_attempts = get_config_value("loop_b.xgc_avis.retry_attempts", 3)

        self.session: Optional[aiohttp.ClientSession] = None
        self.logger.info(f"XGC-AVis client initialized with endpoint: {self.endpoint}")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def estimate_cognitive_state(
        self,
        signals: List[UserSignal],
        context: Optional[Dict[str, Any]] = None
    ) -> CognitiveState:
        """
        Estimate cognitive state from user signals

        Args:
            signals: List of normalized user signals
            context: Optional contextual information

        Returns:
            Estimated CognitiveState
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Prepare request payload
        payload = {
            "signals": [
                {
                    "type": signal.signal_type.value,
                    "value": signal.normalized_value,
                    "timestamp": signal.timestamp.isoformat(),
                    "metadata": signal.metadata
                }
                for signal in signals
            ],
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }

        # Attempt request with retries
        for attempt in range(self.retry_attempts):
            try:
                async with self.session.post(
                    f"{self.endpoint}/estimate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_response(data)
                    else:
                        self.logger.warning(
                            f"XGC-AVis request failed with status {response.status}"
                        )

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"XGC-AVis request timeout (attempt {attempt + 1}/{self.retry_attempts})"
                )
            except aiohttp.ClientError as e:
                self.logger.warning(
                    f"XGC-AVis client error (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                )
            except Exception as e:
                self.logger.error(f"Unexpected error in XGC-AVis request: {e}")

            # Wait before retry
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(0.5 * (attempt + 1))

        # If all attempts failed, return fallback state
        self.logger.error(
            "XGC-AVis estimation failed after all retries. Using fallback."
        )
        return self._get_fallback_state()

    def _parse_response(self, data: Dict[str, Any]) -> CognitiveState:
        """Parse XGC-AVis response into CognitiveState"""
        return CognitiveState(
            cognitive_load=data.get("cognitive_load", 0.5),
            attention_level=data.get("attention_level", 0.5),
            fatigue_index=data.get("fatigue_index", 0.5),
            stress_level=data.get("stress_level", 0.5),
            reading_comprehension=data.get("reading_comprehension", 0.5),
            confidence=data.get("confidence", 0.5),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data else datetime.now()
        )

    def _get_fallback_state(self) -> CognitiveState:
        """Get fallback cognitive state when service is unavailable"""
        return CognitiveState(
            cognitive_load=0.5,
            attention_level=0.5,
            fatigue_index=0.5,
            stress_level=0.5,
            reading_comprehension=0.5,
            confidence=0.0  # Low confidence for fallback
        )

    async def health_check(self) -> bool:
        """
        Check if XGC-AVis service is available

        Returns:
            True if service is healthy, False otherwise
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(
                f"{self.endpoint}/health",
                timeout=aiohttp.ClientTimeout(total=2.0)
            ) as response:
                return response.status == 200
        except Exception as e:
            self.logger.debug(f"XGC-AVis health check failed: {e}")
            return False
