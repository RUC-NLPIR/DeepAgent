"""
Data schemas and models for the Neuroadaptive Accessibility Agent
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class SignalType(str, Enum):
    """Types of input signals from users"""
    EYE_TRACKING = "eye_tracking"
    SPEECH_PATTERNS = "speech_patterns"
    INTERACTION_TIMING = "interaction_timing"
    DEVICE_ORIENTATION = "device_orientation"
    AMBIENT_LIGHT = "ambient_light"
    MOUSE_MOVEMENT = "mouse_movement"
    KEYBOARD_PATTERNS = "keyboard_patterns"


class CognitiveState(BaseModel):
    """Estimated cognitive state of the user"""
    cognitive_load: float = Field(ge=0.0, le=1.0, description="Cognitive load estimate")
    attention_level: float = Field(ge=0.0, le=1.0, description="Attention level")
    fatigue_index: float = Field(ge=0.0, le=1.0, description="Fatigue estimate")
    stress_level: float = Field(ge=0.0, le=1.0, description="Stress level")
    reading_comprehension: float = Field(ge=0.0, le=1.0, description="Reading comprehension estimate")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in estimation")
    timestamp: datetime = Field(default_factory=datetime.now)


class UserSignal(BaseModel):
    """Normalized user signal data"""
    signal_type: SignalType
    raw_value: Any
    normalized_value: float
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AccessibilityAdaptation(BaseModel):
    """Accessibility adaptation recommendation"""
    adaptation_id: str
    category: str
    parameter: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    priority: int = Field(ge=1, le=10, default=5)
    timestamp: datetime = Field(default_factory=datetime.now)


class ContentRefinement(BaseModel):
    """Refined content output from Loop C"""
    original_content: str
    refined_content: str
    refinement_type: Literal["factuality", "personalization", "coherence"]
    changes_made: List[str]
    quality_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    """Memory record for CMS"""
    memory_id: str
    memory_type: Literal["user_preferences", "accessibility_history",
                         "interaction_patterns", "cognitive_profiles"]
    content: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    retention_until: Optional[datetime] = None
    importance_score: float = Field(ge=0.0, le=1.0, default=0.5)


class AgentState(BaseModel):
    """Current state of an agent"""
    agent_id: str
    agent_type: str
    status: Literal["idle", "processing", "waiting", "error", "completed"]
    current_task: Optional[str] = None
    progress: float = Field(ge=0.0, le=1.0, default=0.0)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for Loop E"""
    session_id: str
    adaptation_latency_ms: float
    user_satisfaction_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    accessibility_score: float = Field(ge=0.0, le=1.0)
    refinement_iterations: int
    state_estimation_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    total_adaptations: int
    successful_adaptations: int
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AccessibilityProfile(BaseModel):
    """User accessibility profile"""
    profile_id: str
    profile_name: str
    user_id: Optional[str] = None
    settings: Dict[str, Any]
    cognitive_preferences: Optional[Dict[str, Any]] = None
    sensory_preferences: Optional[Dict[str, Any]] = None
    interaction_preferences: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class AgentMessage(BaseModel):
    """Message passed between agents"""
    message_id: str
    sender_agent: str
    receiver_agent: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: int = Field(ge=1, le=10, default=5)


class LoopStopDecision(BaseModel):
    """Decision from LoopStopChecker"""
    should_stop: bool
    reason: str
    iterations_completed: int
    convergence_score: float = Field(ge=0.0, le=1.0)
    elapsed_time_seconds: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
