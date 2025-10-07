"""
Tests for the flow validator.
"""
import pytest
from datetime import datetime

from ..validator import (
    validate_flow,
    FlowValidationError,
    validate_generated_code
)
from ..schema_models import (
    ConversationFlowOut, NodeOut, EdgeOut, 
    ConversationSettings, EdgePrompt, FunctionSettings,
    STTSettings, LLMSettings, CallSettings, DisplayPosition,
    ElevenLabsTTSSettings
)


def create_basic_flow():
    """Create a basic valid flow for testing"""
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
            temperature=0.7
        ),
        call_settings=CallSettings(
            who_speaks_first="agent",
            end_call_on_silence_ms=30000,
            max_call_duration_ms=600000
        ),
        
        begin_position=DisplayPosition(x=0, y=0),
        start_node_id="node1",
        
        nodes=[
            NodeOut(
                id="node1",
                created=datetime.now(),
                updated=datetime.now(),
                name="Node 1",
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
                from_node_id="node1",
                to_node_id=None,
                type="skip",
                settings=None
            )
        ]
    )


class TestBasicValidation:
    def test_valid_flow_passes(self):
        """Test that a valid flow passes validation"""
        flow = create_basic_flow()
        # Should not raise exception
        validate_flow(flow, strict=False)
    
    def test_missing_start_node_id_fails(self):
        """Test that missing start_node_id fails validation"""
        flow = create_basic_flow()
        flow.start_node_id = None
        
        with pytest.raises(FlowValidationError, match="start_node_id must be set"):
            validate_flow(flow, strict=False)
    
    def test_invalid_start_node_id_fails(self):
        """Test that invalid start_node_id fails validation"""
        flow = create_basic_flow()
        flow.start_node_id = "nonexistent"
        
        with pytest.raises(FlowValidationError, match="start_node_id not found in nodes"):
            validate_flow(flow, strict=False)
    
    def test_duplicate_node_ids_fails(self):
        """Test that duplicate node IDs fail validation"""
        flow = create_basic_flow()
        flow.nodes.append(flow.nodes[0])  # Duplicate the first node
        
        with pytest.raises(FlowValidationError, match="Duplicate node ids found"):
            validate_flow(flow, strict=False)
    
    def test_empty_nodes_fails(self):
        """Test that empty nodes list fails validation"""
        flow = create_basic_flow()
        flow.nodes = []
        
        with pytest.raises(FlowValidationError, match="Flow must have at least one node"):
            validate_flow(flow, strict=False)


class TestNodeValidation:
    def test_conversation_node_missing_settings_fails(self):
        """Test that conversation node without settings fails"""
        flow = create_basic_flow()
        flow.nodes[0].settings = None
        
        with pytest.raises(FlowValidationError, match="missing settings"):
            validate_flow(flow, strict=False)
    
    def test_conversation_node_prompt_without_text_fails(self):
        """Test that prompt type without text fails"""
        flow = create_basic_flow()
        flow.nodes[0].settings.on_enter_type = "prompt"
        flow.nodes[0].settings.on_enter_text = ""
        
        with pytest.raises(FlowValidationError, match="must have on_enter_text"):
            validate_flow(flow, strict=False)
    
    def test_function_node_missing_settings_fails(self):
        """Test that function node without settings fails"""
        flow = create_basic_flow()
        flow.nodes[0].type = "function"
        flow.nodes[0].settings = None

        with pytest.raises(FlowValidationError, match="missing settings"):
            validate_flow(flow, strict=False)


class TestEdgeValidation:
    def test_edge_invalid_from_node_fails(self):
        """Test that edge with invalid from_node_id fails"""
        flow = create_basic_flow()
        flow.edges[0].from_node_id = "nonexistent"
        
        with pytest.raises(FlowValidationError, match="from_node_id not in nodes"):
            validate_flow(flow, strict=False)
    
    def test_edge_invalid_to_node_fails(self):
        """Test that edge with invalid to_node_id fails"""
        flow = create_basic_flow()
        flow.edges[0].to_node_id = "nonexistent"
        flow.edges[0].type = "prompt"
        
        with pytest.raises(FlowValidationError, match="to_node_id not in nodes"):
            validate_flow(flow, strict=False)
    
    def test_prompt_edge_without_to_node_fails(self):
        """Test that prompt edge without to_node_id fails"""
        flow = create_basic_flow()
        flow.edges[0].type = "prompt"
        flow.edges[0].to_node_id = None
        
        with pytest.raises(FlowValidationError, match="prompt edge requires to_node_id"):
            validate_flow(flow, strict=False)
    
    def test_cycle_detection_fails(self):
        """Test that cycles in the flow are detected"""
        flow = create_basic_flow()
        
        # Add second node
        flow.nodes.append(NodeOut(
            id="node2",
            created=datetime.now(),
            updated=datetime.now(),
            name="Node 2",
            is_global=False,
            global_settings=None,
            position=DisplayPosition(x=100, y=0),
            type="conversation",
            settings=ConversationSettings(
                on_enter_text="Hello",
                on_enter_type="prompt",
                allow_interruptions=True,
                skip_response=False,
                finetune_examples=[]
            )
        ))
        
        # Create cycle: node1 -> node2 -> node1
        flow.edges = [
            EdgeOut(
                id="edge1",
                created=datetime.now(),
                updated=datetime.now(),
                from_node_id="node1",
                to_node_id="node2",
                type="prompt",
                settings=EdgePrompt(prompt="Go to node2")
            ),
            EdgeOut(
                id="edge2",
                created=datetime.now(),
                updated=datetime.now(),
                from_node_id="node2",
                to_node_id="node1",
                type="prompt", 
                settings=EdgePrompt(prompt="Go back to node1")
            )
        ]
        
        with pytest.raises(FlowValidationError, match="cycle"):
            validate_flow(flow, strict=False)
    
    def test_no_terminal_path_fails(self):
        """Test that flow without terminal path fails"""
        flow = create_basic_flow()
        
        # Remove terminal edge
        flow.edges = []
        
        with pytest.raises(FlowValidationError, match="No terminal path found"):
            validate_flow(flow, strict=False)


class TestCodeValidation:
    def test_valid_code_passes(self):
        """Test that valid generated code passes validation"""
        code = '''
import logging
from dataclasses import dataclass
from livekit.agents import JobContext
from livekit.plugins import openai

class FlowState:
    pass

class BaseFlowAgent:
    pass

def prewarm(proc):
    pass

async def entrypoint(ctx):
    pass
'''
        
        issues = validate_generated_code(code)
        assert len(issues) == 0
    
    def test_syntax_error_detected(self):
        """Test that syntax errors are detected"""
        code = "def invalid_syntax(:"
        
        issues = validate_generated_code(code)
        assert any("Syntax error" in issue for issue in issues)
    
    def test_missing_imports_detected(self):
        """Test that missing imports are detected"""
        code = "pass"
        
        issues = validate_generated_code(code)
        assert any("Missing required import" in issue for issue in issues)
    
    def test_missing_elements_detected(self):
        """Test that missing required elements are detected"""
        code = '''
import logging
from dataclasses import dataclass
from livekit.agents import JobContext
from livekit.plugins import openai
'''
        
        issues = validate_generated_code(code)
        assert any("Missing required element" in issue for issue in issues)