from datetime import datetime
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field
try:
    # Pydantic v2 style config
    from pydantic import ConfigDict
except Exception:  # pragma: no cover
    ConfigDict = dict  # type: ignore


# --- Providers ---
STT_PROVIDERS = Literal["google", "aws", "azure", "deepgram"]
TTS_PROVIDERS = Literal["aws", "elevenlabs"]
LLM_PROVIDERS = Literal["openai", "azure", "google"]


class CallSettings(BaseModel):
    who_speaks_first: Literal["user", "agent"]
    end_call_on_silence_ms: int
    max_call_duration_ms: int


class STTSettings(BaseModel):
    provider: STT_PROVIDERS
    language: Literal["en-US", "ar-SA"]
    # Optional explicit model for providers that support multiple models (e.g., deepgram "nova-3")
    model: Optional[str] = None
    model_config = ConfigDict(extra="ignore")


class AWSTTSSettings(BaseModel):
    tts_provider: Literal["aws"]
    voice_id: str
    speech_engine: Literal["neural"]


class ElevenLabsTTSSettings(BaseModel):
    class VoiceSettings(BaseModel):
        stability: float
        similarity_boost: float
        style: Optional[float]
        speed: Optional[float]
        use_speaker_boost: Optional[bool]

    tts_provider: Literal["elevenlabs"]
    model: Literal[
        "eleven_multilingual_v2",
        "eleven_flash_v2_5",
        "eleven_flash_v2",
        "eleven_turbo_v2_5",
        "eleven_turbo_v2"
    ]
    voice_id: str
    voice_settings: VoiceSettings
    model_config = ConfigDict(extra="ignore")


TTSSettings = Annotated[
    Union[AWSTTSSettings, ElevenLabsTTSSettings],
    Field(discriminator="tts_provider"),
]


class LLMSettings(BaseModel):
    provider: LLM_PROVIDERS
    # Allow broader model names to support Gemini and future variants while validator enforces per-provider if needed
    model: str
    temperature: float
    max_tokens: Optional[int]
    model_config = ConfigDict(extra="ignore")


class DisplayPosition(BaseModel):
    x: float
    y: float


class GlobalSettings(BaseModel):
    prompt: str
    model_config = ConfigDict(extra="ignore")


class ConversationSettings(BaseModel):
    # Discriminator for settings union
    type: Literal["conversation"] = "conversation"
    class CaptureField(BaseModel):
        name: str
        # primitive types + enum support; lists via multi=true of a primitive
        type: Literal["string", "number", "boolean", "enum"] = "string"
        enum: Optional[list[str]] = None
        multi: Optional[bool] = False
        required: Optional[bool] = False
        description: Optional[str] = None

    on_enter_text: str
    on_enter_type: Literal["prompt", "static"]
    allow_interruptions: bool
    skip_response: bool
    # Optional: declare fields to capture at this node (node-level persistence)
    capture: Optional[list[CaptureField]] = None
    model_config = ConfigDict(extra="ignore")


class EdgePrompt(BaseModel):
    prompt: str
    name: Optional[str] = None


class FunctionSettings(BaseModel):
    """Generic HTTP function node - makes API calls"""
    # Discriminator for settings union
    type: Literal["function"] = "function"
    url: str
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "POST"
    headers: Optional[dict[str, str]] = None
    body: Optional[dict] = None
    timeout_ms: Optional[int] = 10000
    retries: Optional[int] = 0
    wait_for_result: Optional[bool] = True
    class SpeakDuringExecution(BaseModel):
        mode: Literal["prompt", "static"] = "static"
        # If mode == "static", text should be provided
        text: Optional[str] = None
        # If mode == "prompt", instructions should be provided
        instructions: Optional[str] = None
    speak_during_execution: Optional[SpeakDuringExecution] = None
    model_config = ConfigDict(extra="ignore")


class NodeOut(BaseModel):
    id: str
    created: datetime
    updated: datetime

    name: str
    is_global: bool
    global_settings: Optional[GlobalSettings]
    position: DisplayPosition
    type: Literal["conversation", "function"]
    settings: Annotated[Union[ConversationSettings, FunctionSettings], Field(discriminator="type")]
    model_config = ConfigDict(extra="ignore")


class EdgeOut(BaseModel):
    id: str
    created: datetime
    updated: datetime

    from_node_id: str
    to_node_id: Optional[str]
    type: Literal["prompt", "skip"]
    settings: Optional[EdgePrompt]


class ConversationFlowOut(BaseModel):
    id: str
    url_id: str
    created: datetime
    updated: datetime

    name: str
    instructions: str
    stt_settings: STTSettings
    tts_settings: TTSSettings
    llm_settings: LLMSettings
    call_settings: CallSettings
    begin_position: DisplayPosition
    start_node_id: Optional[str]

    nodes: list[NodeOut]
    edges: list[EdgeOut]
    # Ignore unknown top-level fields (e.g., post_call_analysis)
    model_config = ConfigDict(extra="ignore")


