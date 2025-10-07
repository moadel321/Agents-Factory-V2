"""
Tests for the IR (Intermediate Representation) module.
"""
import pytest
from datetime import datetime

from ..ir import build_ir, _classify, _toolify
from ..schema_models import (
    ConversationFlowOut, NodeOut, EdgeOut,
    ConversationSettings, EdgePrompt, FunctionSettings,
    STTSettings, LLMSettings, CallSettings, DisplayPosition,
    ElevenLabsTTSSettings
)


class TestUtilityFunctions:
    """Test utility functions for name transformation"""
    
    def test_classify_simple_name(self):
        """Test classification of simple names"""
        assert _classify("greeting") == "GreetingAgent"
        assert _classify("order") == "OrderAgent"
    
    def test_classify_multi_word_name(self):
        """Test classification of multi-word names"""
        assert _classify("collect_order") == "CollectOrderAgent"
        assert _classify("send_sms") == "SendSmsAgent"
        assert _classify("order_confirmation") == "OrderConfirmationAgent"
    
    def test_classify_special_chars(self):
        """Test classification with special characters"""
        assert _classify("order-confirmation") == "OrderConfirmationAgent"
        assert _classify("send.sms") == "SendSmsAgent"
        assert _classify("node@123") == "Node123Agent"
    
    def test_toolify_simple_name(self):
        """Test toolification of simple names"""
        assert _toolify("proceed") == "go_proceed"
        assert _toolify("continue") == "go_continue"
    
    def test_toolify_multi_word_name(self):
        """Test toolification of multi-word names"""
        assert _toolify("proceed_to_order") == "go_proceed_to_order"
        assert _toolify("send_confirmation") == "go_send_confirmation"
    
    def test_toolify_special_chars(self):
        """Test toolification with special characters"""
        assert _toolify("proceed-to-order") == "go_proceed_to_order"
        assert _toolify("send.confirmation") == "go_send_confirmation"


class TestBasicIRBuild:
    """Test basic IR building functionality"""
    
    def create_simple_flow(self):
        """Create a simple flow for testing"""
        return ConversationFlowOut(
            id="test_flow",
            url_id="test_flow",
            created=datetime.now(),
            updated=datetime.now(),
            name="Test Flow",
            instructions="Test instructions",
            
            stt_settings=STTSettings(provider="google", language="en-US"),
            tts_settings=ElevenLabsTTSSettings(
                tts_provider="elevenlabs",
                model="eleven_flash_v2_5",
                voice_id="test_voice",
                voice_settings=ElevenLabsTTSSettings.VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.5
                )
            ),
            llm_settings=LLMSettings(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1000
            ),
            call_settings=CallSettings(
                who_speaks_first="agent",
                end_call_on_silence_ms=30000,
                max_call_duration_ms=600000
            ),
            
            begin_position=DisplayPosition(x=0, y=0),
            start_node_id="greeting",
            
            nodes=[
                NodeOut(
                    id="greeting",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Greeting",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=0, y=0),
                    type="conversation",
                    settings=ConversationSettings(
                        on_enter_text="Hello, welcome!",
                        on_enter_type="prompt",
                        allow_interruptions=True,
                        skip_response=False,
                        finetune_examples=[]
                    )
                ),
                NodeOut(
                    id="goodbye",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Goodbye",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=200, y=0),
                    type="conversation",
                    settings=ConversationSettings(
                        on_enter_text="Goodbye!",
                        on_enter_type="prompt",
                        allow_interruptions=True,
                        skip_response=False,
                        finetune_examples=[]
                    )
                )
            ],
            
            edges=[
                EdgeOut(
                    id="edge1",
                    created=datetime.now(),
                    updated=datetime.now(),
                    from_node_id="greeting",
                    to_node_id="goodbye",
                    type="prompt",
                    settings=EdgePrompt(
                        prompt="Proceed to goodbye",
                        name="say_goodbye"
                    )
                ),
                EdgeOut(
                    id="edge2",
                    created=datetime.now(),
                    updated=datetime.now(),
                    from_node_id="goodbye",
                    to_node_id=None,
                    type="skip",
                    settings=None
                )
            ]
        )
    
    def test_basic_ir_structure(self):
        """Test that basic IR structure is correct"""
        flow = self.create_simple_flow()
        ir = build_ir(flow)
        
        # Check basic properties
        assert ir.url_id == "test_flow"
        assert ir.name == "Test Flow"
        assert ir.instructions == "Test instructions"
        assert ir.stt_provider == "google"
        assert ir.start_class_name == "GreetingAgent"
        
        # Check LLM config
        assert ir.llm["model"] == "gpt-4o-mini"
        assert ir.llm["temperature"] == 0.7
        assert ir.llm["max_tokens"] == 1000
        
        # Check TTS config
        assert ir.tts["voice_id"] == "test_voice"
        
        # Check nodes
        assert len(ir.nodes) == 2
        assert ir.nodes[0]["id"] == "greeting"
        assert ir.nodes[0]["class_name"] == "GreetingAgent"
        assert ir.nodes[0]["type"] == "conversation"
    
    
    def test_edge_tools_generation(self):
        """Test that edge tools are correctly generated"""
        flow = self.create_simple_flow()
        ir = build_ir(flow)
        
        # Check first node's outgoing edges
        greeting_node = ir.nodes[0]
        assert len(greeting_node["out_edges"]) == 1
        
        edge = greeting_node["out_edges"][0]
        assert edge["tool_name"] == "go_say_goodbye"
        assert edge["description"] == "Proceed to goodbye"
        assert edge["next_class_name"] == "GoodbyeAgent"
        
        # Check second node (terminal)
        goodbye_node = ir.nodes[1]
        assert len(goodbye_node["out_edges"]) == 1
        
        terminal_edge = goodbye_node["out_edges"][0]
        assert terminal_edge["tool_name"] == "end_conversation"
        assert terminal_edge["description"] == "End the conversation"
        assert terminal_edge["next_class_name"] is None


class TestFunctionNodeIR:
    """Test IR building for function nodes"""
    
    def create_function_flow(self):
        """Create a flow with function nodes for testing"""
        return ConversationFlowOut(
            id="test_function_flow",
            url_id="test_function_flow",
            created=datetime.now(),
            updated=datetime.now(),
            name="Function Test Flow",
            instructions="Test function flow",
            
            stt_settings=STTSettings(provider="google", language="en-US"),
            tts_settings=ElevenLabsTTSSettings(
                tts_provider="elevenlabs",
                model="eleven_flash_v2_5",
                voice_id="test_voice",
                voice_settings=ElevenLabsTTSSettings.VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.5
                )
            ),
            llm_settings=LLMSettings(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.7
            ),
            call_settings=CallSettings(
                who_speaks_first="agent",
                end_call_on_silence_ms=30000,
                max_call_duration_ms=600000
            ),
            
            begin_position=DisplayPosition(x=0, y=0),
            start_node_id="sms_node",
            
            nodes=[
                NodeOut(
                    id="sms_node",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Send SMS",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=0, y=0),
                    type="function",
                    settings=FunctionSettings(
                        url="https://api.example.com/sms/send",
                        method="POST",
                        headers={"Content-Type": "application/json"},
                        body={
                            "to": "{phone}",
                            "message": "{message}"
                        },
                        timeout_ms=15000,
                        retries=3
                    )
                )
            ],
            
            edges=[
                EdgeOut(
                    id="edge1",
                    created=datetime.now(),
                    updated=datetime.now(),
                    from_node_id="sms_node",
                    to_node_id=None,
                    type="skip",
                    settings=None
                )
            ]
        )
    
    def test_function_node_ir(self):
        """Test IR generation for function nodes"""
        flow = self.create_function_flow()
        ir = build_ir(flow)

        # Check node IR
        sms_node = ir.nodes[0]
        assert sms_node["type"] == "function"
        assert sms_node["url"] == "https://api.example.com/sms/send"
        assert sms_node["method"] == "POST"
        assert sms_node["headers"] == {"Content-Type": "application/json"}
        assert sms_node["body"] == {"to": "{phone}", "message": "{message}"}
        assert sms_node["timeout_ms"] == 15000
        assert sms_node["retries"] == 3


class TestCallSettingsIR:
    """Test IR building for call settings"""
    
    def test_call_settings_ir(self):
        """Test IR generation for call settings"""
        flow = ConversationFlowOut(
            id="test_call_settings_flow",
            url_id="test_call_settings_flow",
            created=datetime.now(),
            updated=datetime.now(),
            name="Call Settings Test Flow",
            instructions="Test call settings flow",
            
            stt_settings=STTSettings(provider="google", language="en-US"),
            tts_settings=ElevenLabsTTSSettings(
                tts_provider="elevenlabs",
                model="eleven_flash_v2_5",
                voice_id="test_voice",
                voice_settings=ElevenLabsTTSSettings.VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.5
                )
            ),
            llm_settings=LLMSettings(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.7
            ),
            call_settings=CallSettings(
                who_speaks_first="user",
                end_call_on_silence_ms=45000,
                max_call_duration_ms=900000
            ),
            
            begin_position=DisplayPosition(x=0, y=0),
            start_node_id="greeting",
            
            nodes=[
                NodeOut(
                    id="greeting",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Greeting",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=0, y=0),
                    type="conversation",
                    settings=ConversationSettings(
                        on_enter_text="Hello",
                        on_enter_type="prompt",
                        allow_interruptions=True,
                        skip_response=False,
                        finetune_examples=[]
                    )
                )
            ],
            
            edges=[
                EdgeOut(
                    id="edge1",
                    created=datetime.now(),
                    updated=datetime.now(),
                    from_node_id="greeting",
                    to_node_id=None,
                    type="skip",
                    settings=None
                )
            ]
        )
        
        ir = build_ir(flow)
        
        # Check call settings
        assert ir.call_settings is not None
        assert ir.call_settings.who_speaks_first == "user"
        assert ir.call_settings.end_call_on_silence_ms == 45000
        assert ir.call_settings.max_call_duration_ms == 900000