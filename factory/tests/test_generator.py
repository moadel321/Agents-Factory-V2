"""
Tests for the code generator.
"""
import pytest
import tempfile
import os
from datetime import datetime

from ..generator import CodeGenerator, generate_from_json
from ..schema_models import (
    ConversationFlowOut, NodeOut, EdgeOut,
    ConversationSettings, EdgePrompt, FunctionSettings,
    STTSettings, LLMSettings, CallSettings, DisplayPosition,
    ElevenLabsTTSSettings
)


class TestCodeGenerator:
    """Test the main CodeGenerator class"""
    
    def create_simple_flow(self):
        """Create a simple flow for testing"""
        return ConversationFlowOut(
            id="test_flow",
            url_id="simple_test",
            created=datetime.now(),
            updated=datetime.now(),
            name="Simple Test Flow",
            instructions="You are a helpful assistant.",
            
            stt_settings=STTSettings(provider="google", language="en-US"),
            tts_settings=ElevenLabsTTSSettings(
                tts_provider="elevenlabs",
                model="eleven_flash_v2_5",
                voice_id="21m00Tcm4TlvDq8ikWAM",
                voice_settings=ElevenLabsTTSSettings.VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75
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
            start_node_id="greeting",
            
            nodes=[
                NodeOut(
                    id="greeting",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Greeting Node",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=0, y=0),
                    type="conversation",
                    settings=ConversationSettings(
                        on_enter_text="Hello! How can I help you today?",
                        on_enter_type="prompt",
                        allow_interruptions=True,
                        skip_response=False,
                        finetune_examples=[]
                    )
                ),
                NodeOut(
                    id="farewell",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Farewell Node",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=200, y=0),
                    type="conversation",
                    settings=ConversationSettings(
                        on_enter_text="Thank you! Have a great day!",
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
                    to_node_id="farewell",
                    type="prompt",
                    settings=EdgePrompt(
                        prompt="User wants to end the conversation",
                        name="proceed_to_farewell"
                    )
                ),
                EdgeOut(
                    id="edge2",
                    created=datetime.now(),
                    updated=datetime.now(),
                    from_node_id="farewell",
                    to_node_id=None,
                    type="skip",
                    settings=None
                )
            ]
        )
    
    def test_generator_initialization(self):
        """Test generator initialization"""
        generator = CodeGenerator()
        assert generator.env is not None
        
        # Test custom filters are registered
        assert 'classify' in generator.env.filters
        assert 'toolify' in generator.env.filters
        assert 'jsonify' in generator.env.filters
    
    def test_classify_filter(self):
        """Test the classify filter"""
        generator = CodeGenerator()
        classify = generator.env.filters['classify']
        
        assert classify("greeting") == "GreetingAgent"
        assert classify("order_summary") == "OrderSummaryAgent"
        assert classify("send-sms") == "SendSmsAgent"
    
    def test_toolify_filter(self):
        """Test the toolify filter"""
        generator = CodeGenerator()
        toolify = generator.env.filters['toolify']
        
        assert toolify("proceed") == "go_proceed"
        assert toolify("send_confirmation") == "go_send_confirmation"
        assert toolify("end-call") == "go_end_call"
    
    def test_generate_agent_basic(self):
        """Test basic agent generation"""
        generator = CodeGenerator()
        flow = self.create_simple_flow()
        
        code = generator.generate_agent(flow, validate=False)
        
        # Check basic structure
        assert "class FlowState" in code
        assert "class BaseFlowAgent" in code
        assert "class GreetingNodeAgent" in code
        assert "class FarewellNodeAgent" in code
        assert "def entrypoint" in code
        assert "def prewarm" in code
        
        # Check specific elements
        assert "simple_test" in code  # URL ID
        assert "You are a helpful assistant" in code  # Instructions
        assert "Hello! How can I help you today?" in code  # On-enter text
        assert "go_proceed_to_farewell" in code  # Tool name
    
    def test_generate_agent_with_file_output(self):
        """Test generating agent to file"""
        generator = CodeGenerator()
        flow = self.create_simple_flow()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
            try:
                code = generator.generate_agent(flow, tmp_file.name, validate=False)
                
                # Check file was created
                assert os.path.exists(tmp_file.name)
                
                # Check file contents
                with open(tmp_file.name, 'r') as f:
                    file_code = f.read()
                
                assert file_code == code
                assert "class FlowState" in file_code
                
            finally:
                os.unlink(tmp_file.name)
    
    def test_generate_with_validation_error(self):
        """Test generation with validation error"""
        generator = CodeGenerator()
        flow = self.create_simple_flow()
        
        # Create invalid flow (no start_node_id)
        flow.start_node_id = None
        
        with pytest.raises(Exception):  # Should raise validation error
            generator.generate_agent(flow, validate=True)
    
    def test_template_context_building(self):
        """Test template context building"""
        generator = CodeGenerator()
        flow = self.create_simple_flow()
        
        from ..ir import build_ir
        ir_data = build_ir(flow)
        context = generator._build_template_context(flow, ir_data)
        
        # Check context structure
        assert "flow" in context
        flow_context = context["flow"]
        
        assert flow_context["url_id"] == "simple_test"
        assert flow_context["name"] == "Simple Test Flow"
        assert flow_context["instructions"] == "You are a helpful assistant."
        assert flow_context["stt_provider"] == "google"
        assert "llm" in flow_context
        assert "tts" in flow_context
        assert "nodes" in flow_context
        assert flow_context["start_class_name"] == "GreetingNodeAgent"
    
    def _build_ir_for_context(self, flow):
        """Helper to build IR for context testing"""
        from ..ir import build_ir
        return build_ir(flow)


class TestFunctionNodeGeneration:
    """Test generation for function nodes"""
    
    def create_function_flow(self):
        """Create a flow with function nodes"""
        return ConversationFlowOut(
            id="function_test_flow",
            url_id="function_test",
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
            start_node_id="collect_info",
            
            nodes=[
                NodeOut(
                    id="collect_info",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Collect Info",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=0, y=0),
                    type="conversation",
                    settings=ConversationSettings(
                        on_enter_text="Please provide your phone number and message.",
                        on_enter_type="prompt",
                        allow_interruptions=True,
                        skip_response=False,
                        finetune_examples=[]
                    )
                ),
                NodeOut(
                    id="send_sms",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Send SMS",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=200, y=0),
                    type="function",
                    settings=FunctionSettings(
                        url="https://api.example.com/sms/send",
                        method="POST",
                        headers={"Content-Type": "application/json"},
                        body={
                            "to": "{phone}",
                            "message": "{message}"
                        },
                        timeout_ms=10000,
                        retries=2
                    )
                ),
                NodeOut(
                    id="confirmation",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Confirmation",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=400, y=0),
                    type="conversation",
                    settings=ConversationSettings(
                        on_enter_text="SMS sent successfully!",
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
                    from_node_id="collect_info",
                    to_node_id="send_sms",
                    type="prompt",
                    settings=EdgePrompt(
                        prompt="User provided phone and message",
                        name="proceed_to_sms"
                    )
                ),
                EdgeOut(
                    id="edge2",
                    created=datetime.now(),
                    updated=datetime.now(),
                    from_node_id="send_sms",
                    to_node_id="confirmation",
                    type="prompt",
                    settings=EdgePrompt(
                        prompt="SMS task completed",
                        name="proceed_to_confirmation"
                    )
                ),
                EdgeOut(
                    id="edge3",
                    created=datetime.now(),
                    updated=datetime.now(),
                    from_node_id="confirmation",
                    to_node_id=None,
                    type="skip",
                    settings=None
                )
            ]
        )
    
    def test_function_node_generation(self):
        """Test generation of function nodes"""
        generator = CodeGenerator()
        flow = self.create_function_flow()

        code = generator.generate_agent(flow, validate=False)

        # Check function node class exists
        assert "class SendSmsAgent" in code

        # Check generic HTTP function execution method
        assert "_execute_function_task" in code
        assert "https://api.example.com/sms/send" in code

        # Check HTTP request configuration is included
        assert "timeout_ms" in code or "10000" in code
        assert "retries" in code or "2" in code


class TestGenerateFromJSON:
    """Test convenience functions for JSON generation"""
    
    def test_generate_from_json_string(self):
        """Test generating from JSON string"""
        flow_json = '''
        {
            "id": "json_test_flow",
            "url_id": "json_test",
            "created": "2024-01-01T00:00:00",
            "updated": "2024-01-01T00:00:00",
            "name": "JSON Test Flow",
            "instructions": "Test from JSON",
            "stt_settings": {
                "provider": "google",
                "language": "en-US"
            },
            "tts_settings": {
                "tts_provider": "elevenlabs",
                "model": "eleven_flash_v2_5",
                "voice_id": "test_voice",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            },
            "llm_settings": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.7
            },
            "call_settings": {
                "who_speaks_first": "agent",
                "end_call_on_silence_ms": 30000,
                "max_call_duration_ms": 600000
            },
            "begin_position": {"x": 0, "y": 0},
            "start_node_id": "test_node",
            "nodes": [
                {
                    "id": "test_node",
                    "created": "2024-01-01T00:00:00",
                    "updated": "2024-01-01T00:00:00",
                    "name": "Test Node",
                    "is_global": false,
                    "global_settings": null,
                    "position": {"x": 0, "y": 0},
                    "type": "conversation",
                    "settings": {
                        "on_enter_text": "Hello from JSON!",
                        "on_enter_type": "prompt",
                        "allow_interruptions": true,
                        "skip_response": false,
                        "finetune_examples": []
                    }
                }
            ],
            "edges": [
                {
                    "id": "edge1",
                    "created": "2024-01-01T00:00:00",
                    "updated": "2024-01-01T00:00:00",
                    "from_node_id": "test_node",
                    "to_node_id": null,
                    "type": "skip",
                    "settings": null
                }
            ]
        }
        '''
        
        code = generate_from_json(flow_json, validate=False)
        
        # Check generated code
        assert "Hello from JSON!" in code
        assert "json_test" in code  # URL ID
        assert "class TestNodeAgent" in code
        assert "Test from JSON" in code  # Instructions
    
    def test_generate_from_json_with_file_output(self):
        """Test generating from JSON with file output"""
        flow_json = '''
        {
            "id": "json_file_test_flow",
            "url_id": "json_file_test",
            "created": "2024-01-01T00:00:00",
            "updated": "2024-01-01T00:00:00",
            "name": "JSON File Test Flow",
            "instructions": "Test from JSON to file",
            "stt_settings": {"provider": "google", "language": "en-US"},
            "tts_settings": {
                "tts_provider": "elevenlabs",
                "model": "eleven_flash_v2_5",
                "voice_id": "test_voice",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
            },
            "llm_settings": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.7},
            "call_settings": {"who_speaks_first": "agent", "end_call_on_silence_ms": 30000, "max_call_duration_ms": 600000},
            "begin_position": {"x": 0, "y": 0},
            "start_node_id": "test_node",
            "nodes": [
                {
                    "id": "test_node",
                    "created": "2024-01-01T00:00:00",
                    "updated": "2024-01-01T00:00:00",
                    "name": "Test Node",
                    "is_global": false,
                    "global_settings": null,
                    "position": {"x": 0, "y": 0},
                    "type": "conversation",
                    "settings": {
                        "on_enter_text": "Hello!",
                        "on_enter_type": "prompt",
                        "allow_interruptions": true,
                        "skip_response": false,
                        "finetune_examples": []
                    }
                }
            ],
            "edges": [
                {
                    "id": "edge1",
                    "created": "2024-01-01T00:00:00",
                    "updated": "2024-01-01T00:00:00",
                    "from_node_id": "test_node",
                    "to_node_id": null,
                    "type": "skip",
                    "settings": null
                }
            ]
        }
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
            try:
                code = generate_from_json(flow_json, tmp_file.name, validate=False)
                
                # Check file was created
                assert os.path.exists(tmp_file.name)
                
                # Check file contents match returned code
                with open(tmp_file.name, 'r') as f:
                    file_code = f.read()
                
                assert file_code == code
                
            finally:
                os.unlink(tmp_file.name)