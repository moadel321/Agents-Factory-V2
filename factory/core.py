"""
Core models and base classes for the flow agent factory.
"""
import os
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from livekit.agents import JobContext
from livekit.agents.voice import Agent
from livekit.plugins import deepgram, openai, elevenlabs, silero
from livekit import api

logger = logging.getLogger(__name__)


@dataclass
class FlowState:
    """
    Central state management for conversation flows.
    Bound to session.userdata to persist across agent transitions.
    """
    # Generic data storage for collected information
    slots: Dict[str, Any] = field(default_factory=dict)
    
    # Path tracking - list of node IDs visited
    path: List[str] = field(default_factory=list)
    
    # Results from function node tasks
    task_results: Dict[str, Any] = field(default_factory=dict)
    
    def add_to_path(self, node_id: str) -> None:
        """Add a node to the conversation path"""
        self.path.append(node_id)
        logger.debug(f"Added {node_id} to path: {' -> '.join(self.path)}")
        
    def set_slot(self, key: str, value: Any) -> None:
        """Store data in a named slot"""
        self.slots[key] = value
        logger.debug(f"Set slot {key} = {value}")
        
    def get_slot(self, key: str, default: Any = None) -> Any:
        """Retrieve data from a named slot"""
        return self.slots.get(key, default)
        
    def has_slot(self, key: str) -> bool:
        """Check if a slot exists"""
        return key in self.slots
        
    def remove_slot(self, key: str) -> Any:
        """Remove and return a slot value"""
        return self.slots.pop(key, None)
        
    def clear_slots(self) -> None:
        """Clear all slot data"""
        self.slots.clear()
        
    def get_path_string(self) -> str:
        """Get the path as a formatted string"""
        return " -> ".join(self.path)


class BaseFlowAgent(Agent):
    """
    Base agent class with common functionality for all flow nodes.
    Centralizes plugin initialization, utilities, and error handling.
    """
    
    def __init__(
        self, 
        job_context: JobContext, 
        instructions: str,
        stt_provider: str = "deepgram",
        llm_config: Optional[Dict[str, Any]] = None,
        tts_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.job_context = job_context
        
        # Default configurations
        llm_config = llm_config or {"model": "gpt-4o-mini", "temperature": 0.7}
        tts_config = tts_config or {"model": "eleven_multilingual_v2"}
        
        # Initialize STT based on provider
        if stt_provider == "google":
            stt = deepgram.STT(model="nova-2")  # Fallback to deepgram for now
        elif stt_provider == "aws":
            stt = deepgram.STT(model="nova-2")  # Fallback to deepgram for now
        else:  # Default to deepgram
            stt = deepgram.STT(model="nova-2", language="multi")
            
        # Initialize LLM (OpenAI only as per requirements)
        llm = openai.LLM(
            model=llm_config.get("model", "gpt-4o-mini"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens")
        )
        
        # Initialize TTS (ElevenLabs as per requirements)
        eleven_api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
        if tts_config.get("voice_id"):
            tts = elevenlabs.TTS(
                api_key=eleven_api_key,
                model=tts_config.get("model", "eleven_multilingual_v2"),
                voice=tts_config["voice_id"]
            )
        else:
            tts = elevenlabs.TTS(
                api_key=eleven_api_key,
                model=tts_config.get("model", "eleven_multilingual_v2")
            )
        
        super().__init__(
            instructions=instructions,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=silero.VAD.load()  # VAD prewarm as per requirements
        )
    
    async def say_or_skip(self, text: Optional[str], skip_response: bool = False) -> None:
        """
        Utility to conditionally speak text.
        
        Args:
            text: Text to speak
            skip_response: If True, don't speak even if text is provided
        """
        if text and not skip_response:
            await self.session.say(text)
            logger.debug(f"Agent said: {text}")
        elif skip_response:
            logger.debug(f"Skipped saying: {text}")
    
    async def end_call_if_needed(self) -> None:
        """
        End the call and perform cleanup.
        Closes the session and optionally deletes the room.
        """
        try:
            logger.info("Ending call and cleaning up")
            await self.session.aclose()
            
            # Delete room if requested
            if os.getenv("DELETE_ROOM_ON_END", "true").lower() == "true":
                request = api.DeleteRoomRequest(room=self.job_context.room.name)
                await self.job_context.api.room.delete_room(request)
                logger.info("Room deleted successfully")
        except Exception as e:
            logger.error(f"Error during call cleanup: {e}")
    
    def get_flow_state(self) -> FlowState:
        """Get the current flow state from session userdata"""
        if not hasattr(self.session, 'userdata') or self.session.userdata is None:
            logger.warning("No FlowState found in session, creating new one")
            self.session.userdata = FlowState()
        return self.session.userdata
    
    async def handle_silence_timeout(self, duration_ms: int = 30000) -> None:
        """
        Handle silence timeout - can be overridden by subclasses.
        
        Args:
            duration_ms: Timeout duration in milliseconds
        """
        logger.warning(f"Silence timeout after {duration_ms}ms")
        await self.session.say("I haven't heard from you in a while. Let me end this call.")
        await self.end_call_if_needed()
    
    async def handle_call_duration_limit(self, max_duration_ms: int = 600000) -> None:
        """
        Handle maximum call duration limit - can be overridden by subclasses.
        
        Args:
            max_duration_ms: Maximum call duration in milliseconds
        """
        logger.warning(f"Call duration limit reached: {max_duration_ms}ms")
        await self.session.say("We've reached the maximum call duration. Thank you for your time.")
        await self.end_call_if_needed()
    
    async def log_interaction(self, interaction_type: str, data: Dict[str, Any]) -> None:
        """
        Log interaction data for analytics/debugging.
        
        Args:
            interaction_type: Type of interaction (e.g., "tool_call", "response")
            data: Interaction data to log
        """
        flow_state = self.get_flow_state()
        log_entry = {
            "timestamp": asyncio.get_event_loop().time(),
            "room": self.job_context.room.name,
            "path": flow_state.get_path_string(),
            "type": interaction_type,
            "data": data
        }
        logger.info(f"Interaction: {log_entry}")