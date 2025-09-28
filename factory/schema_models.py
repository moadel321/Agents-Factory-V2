from datetime import datetime
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


# --- Providers ---
STT_PROVIDERS = Literal["google", "aws", "azure", "deepgram"]
TTS_PROVIDERS = Literal["aws", "elevenlabs"]
LLM_PROVIDERS = Literal["openai", "azure", "google"]
POST_CALL_ANALYSIS_TYPES = Literal["boolean", "text", "number", "selector"]


class CallSettings(BaseModel):
    who_speaks_first: Literal["user", "agent"]
    end_call_on_silence_ms: int
    max_call_duration_ms: int


class STTSettings(BaseModel):
    provider: STT_PROVIDERS
    language: Literal["en-US", "ar-SA"]
    # Optional explicit model for providers that support multiple models (e.g., deepgram "nova-3")
    model: Optional[str] = None


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


class PostCallAnalysisItem(BaseModel):
    name: str
    description: str
    type: POST_CALL_ANALYSIS_TYPES
    selector_options: Optional[list[str]]


class PostCallAnalysisSettings(BaseModel):
    model: Literal["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]
    analysis_items: list[PostCallAnalysisItem]


class DisplayPosition(BaseModel):
    x: float
    y: float


class LLMSimpleOverrides(BaseModel):
    provider: Optional[LLM_PROVIDERS]
    model: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]


class GlobalSettings(BaseModel):
    class FinetuneExample(BaseModel):
        class ConversationExample(BaseModel):
            speaker: Literal["user", "agent"]
            text: str

        type: Literal["jump", "not_jump"]
        conversations: list[ConversationExample]

    prompt: str
    finetune_examples: list[FinetuneExample]


class ConversationSettings(BaseModel):
    class CaptureField(BaseModel):
        name: str
        # primitive types + enum support; lists via multi=true of a primitive
        type: Literal["string", "number", "boolean", "enum"] = "string"
        enum: Optional[list[str]] = None
        multi: Optional[bool] = False
        required: Optional[bool] = False
        description: Optional[str] = None

    class FinetuneExample(BaseModel):
        class ConversationExample(BaseModel):
            speaker: Literal["user", "agent"]
            text: str

        type: Literal["conversation"]
        conversations: list[ConversationExample]

    on_enter_text: str
    on_enter_type: Literal["prompt", "static"]
    allow_interruptions: bool
    skip_response: bool
    finetune_examples: list[FinetuneExample]
    llm_overrides: Optional[LLMSimpleOverrides]
    # Optional: declare fields to capture at this node (node-level persistence)
    capture: Optional[list[CaptureField]] = None


class EdgePrompt(BaseModel):
    prompt: str
    name: Optional[str] = None


class FunctionSettings(BaseModel):
    function_type: Literal["sms", "call_transfer", "rest_webhook"]
    parameters_schema: Optional[dict]
    timeout_ms: Optional[int] = 10000
    retries: Optional[int] = 0


class NodeOut(BaseModel):
    id: str
    created: datetime
    updated: datetime

    name: str
    is_global: bool
    global_settings: Optional[GlobalSettings]
    position: DisplayPosition
    type: Literal["conversation", "function"]
    settings: ConversationSettings
    function: Optional[FunctionSettings] = None


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
    post_call_analysis: Optional[PostCallAnalysisSettings]
    begin_position: DisplayPosition
    start_node_id: Optional[str]

    nodes: list[NodeOut]
    edges: list[EdgeOut]


