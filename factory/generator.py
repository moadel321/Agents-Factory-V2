"""
Main code generator for flow agents using Jinja2 templates.
"""
import os
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from .schema_models import ConversationFlowOut
from .ir import IRFlow, build_ir
from .validator import validate_flow, FlowValidationError

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Main code generator class"""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the code generator.
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        if template_dir is None:
            # Default to templates directory next to this file
            factory_dir = Path(__file__).parent
            template_dir = str(factory_dir / "templates")
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=False,
            lstrip_blocks=False
        )
        
        # Add custom filters
        self.env.filters['classify'] = self._classify
        self.env.filters['toolify'] = self._toolify
        self.env.filters['ascii_slug'] = self._ascii_slug
        self.env.filters['jsonify'] = json.dumps
        
    @staticmethod
    def _classify(name: str) -> str:
        """Convert a name to a Python class name"""
        safe = "".join(ch if ch.isalnum() else "_" for ch in name)
        parts = [p for p in safe.split("_") if p]
        return "".join(s.capitalize() for s in parts) + "Agent"
    
    @staticmethod
    def _toolify(name: str) -> str:
        """Convert a display name to an ASCII-safe tool name"""
        slug = CodeGenerator._ascii_slug(name)
        # Convert to valid Python method identifier (no hyphens)
        slug = (slug if slug else "tool").replace('-', '_')
        tool_name = "go_" + slug
        return CodeGenerator._cap_identifier(tool_name)

    @staticmethod
    def _ascii_slug(name: str) -> str:
        import unicodedata
        normalized = unicodedata.normalize('NFKD', name or '')
        ascii_str = normalized.encode('ascii', 'ignore').decode('ascii')
        safe = ''.join(ch if (ch.isalnum() or ch in '-_') else '_' for ch in ascii_str)
        parts = [p for p in safe.split('_') if p]
        slug = '_'.join(parts)
        if slug and slug[0].isdigit():
            slug = f"n_{slug}"
        return slug

    @staticmethod
    def _cap_identifier(identifier: str, max_len: int = 64) -> str:
        """Cap identifier length using slicing with short hash suffix for uniqueness."""
        if len(identifier) <= max_len:
            return identifier
        import hashlib
        digest = hashlib.sha1(identifier.encode('utf-8')).hexdigest()[:4]
        head = identifier[: max_len - 5]
        return f"{head}_{digest}"
    
    def generate_agent(
        self,
        flow: ConversationFlowOut,
        output_path: Optional[str] = None,
        validate: bool = True,
        format_code: bool = True
    ) -> str:
        """
        Generate agent code from a flow definition.

        Args:
            flow: Flow definition
            output_path: Where to save the generated file (optional)
            validate: Whether to validate the flow first
            format_code: Whether to format the generated code with ruff

        Returns:
            Generated Python code as string

        Raises:
            FlowValidationError: If flow validation fails
        """
        # Validate flow if requested
        if validate:
            try:
                validate_flow(flow)
                logger.info("Flow validation passed")
            except FlowValidationError as e:
                logger.error(f"Flow validation failed: {e}")
                raise
        
        # Convert to intermediate representation
        ir = build_ir(flow)
        logger.info(f"Built IR with {len(ir.nodes)} nodes, starting with {ir.start_class_name}")
        
        # Add additional context for template
        template_context = self._build_template_context(flow, ir)
        
        # Load and render template
        template = self.env.get_template("agent.jinja2")
        code = template.render(**template_context)
        
        # Save to file if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Generated agent saved to {output_path}")

            # Format the generated code
            if format_code:
                self._format_code(output_path)

        return code

    def _format_code(self, file_path: str) -> None:
        """
        Format generated code using ruff.

        Args:
            file_path: Path to the file to format
        """
        if not shutil.which("ruff"):
            logger.warning("ruff not found in PATH, skipping formatting")
            return

        try:
            # Run ruff check --fix to auto-fix issues
            result = subprocess.run(
                ["ruff", "check", "--fix", file_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30
            )
            if result.returncode != 0:
                logger.warning(f"ruff check --fix warnings: {result.stderr}")

            # Run ruff format to format code
            result = subprocess.run(
                ["ruff", "format", file_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30
            )
            if result.returncode == 0:
                logger.info(f"Formatted {file_path} with ruff")
            else:
                logger.warning(f"ruff format failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.warning(f"ruff formatting timed out for {file_path}")
        except Exception as e:
            logger.warning(f"ruff formatting failed for {file_path}: {e}")

    def _build_template_context(self, flow: ConversationFlowOut, ir: IRFlow) -> Dict[str, Any]:
        """
        Build context dictionary for template rendering.
        
        Args:
            flow: Original flow definition
            ir: Intermediate representation
            
        Returns:
            Template context dictionary
        """
        context = {
            "flow": {
                "url_id": flow.url_id,
                "name": flow.name,
                "instructions": flow.instructions,
                # Provider settings (normalized for template)
                "stt": {
                    "provider": flow.stt_settings.provider,
                    "language": flow.stt_settings.language,
                    "model": getattr(flow.stt_settings, "model", None),
                },
                "llm": {
                    "provider": flow.llm_settings.provider,
                    "model": flow.llm_settings.model,
                    "temperature": flow.llm_settings.temperature,
                    "max_tokens": flow.llm_settings.max_tokens,
                },
                "tts": (lambda ts: {
                    "provider": getattr(ts, "tts_provider", None),
                    "model": getattr(ts, "model", None),
                    "voice_id": getattr(ts, "voice_id", None),
                    "speech_engine": getattr(ts, "speech_engine", None),
                })(flow.tts_settings),
                "nodes": ir.nodes,
                "start_class_name": ir.start_class_name
            }
        }

        # Add call settings if available
        if flow.call_settings:
            context["flow"]["call_settings"] = {
                "who_speaks_first": flow.call_settings.who_speaks_first,
                "end_call_on_silence_ms": flow.call_settings.end_call_on_silence_ms,
                "max_call_duration_ms": flow.call_settings.max_call_duration_ms
            }

        return context
    
    def generate_multiple_agents(
        self, 
        flows: list[ConversationFlowOut], 
        output_dir: str = "generated",
        validate: bool = True
    ) -> Dict[str, str]:
        """
        Generate multiple agents from a list of flows.
        
        Args:
            flows: List of flow definitions
            output_dir: Directory to save generated files
            validate: Whether to validate each flow
            
        Returns:
            Dict mapping flow URL IDs to generated file paths
        """
        results = {}
        
        for flow in flows:
            try:
                output_path = os.path.join(output_dir, f"agent_{flow.url_id}.py")
                self.generate_agent(flow, output_path, validate)
                results[flow.url_id] = output_path
                logger.info(f"Successfully generated agent for flow {flow.url_id}")
            except Exception as e:
                logger.error(f"Failed to generate agent for flow {flow.url_id}: {e}")
                results[flow.url_id] = f"ERROR: {str(e)}"
        
        return results


def generate_from_json(
    flow_json: str,
    output_path: Optional[str] = None,
    validate: bool = True
) -> str:
    """
    Convenience function to generate agent from JSON string.
    
    Args:
        flow_json: JSON string containing flow definition
        output_path: Where to save the generated file
        validate: Whether to validate the flow
        
    Returns:
        Generated Python code
    """
    flow_data = json.loads(flow_json)
    flow = ConversationFlowOut(**flow_data)
    
    generator = CodeGenerator()
    return generator.generate_agent(flow, output_path, validate)


def generate_from_file(
    flow_file_path: str,
    output_path: Optional[str] = None,
    validate: bool = True
) -> str:
    """
    Convenience function to generate agent from JSON file.
    
    Args:
        flow_file_path: Path to JSON file containing flow definition
        output_path: Where to save the generated file
        validate: Whether to validate the flow
        
    Returns:
        Generated Python code
    """
    with open(flow_file_path, 'r', encoding='utf-8') as f:
        flow_json = f.read()
    
    return generate_from_json(flow_json, output_path, validate)


# Example usage for testing
if __name__ == "__main__":
    # Create a simple test flow
    from datetime import datetime
    from .schema_models import (
        NodeOut, EdgeOut, ConversationSettings, EdgePrompt,
        STTSettings, LLMSettings, CallSettings, DisplayPosition,
        ElevenLabsTTSSettings
    )
    
    # Test flow definition
    test_flow = ConversationFlowOut(
        id="test_flow_123",
        url_id="test_pizza_flow",
        created=datetime.now(),
        updated=datetime.now(),
        name="Pizza Ordering Flow",
        instructions="You are a helpful pizza ordering assistant.",
        stt_settings=STTSettings(provider="google", language="en-US"),
        tts_settings=ElevenLabsTTSSettings(
            tts_provider="elevenlabs",
            model="eleven_monolingual_v1",
            voice_id="21m00Tcm4TlvDq8ikWAM",
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
                    on_enter_text="Hi! Welcome to Pizza Palace. What would you like to order?",
                    on_enter_type="prompt",
                    allow_interruptions=True,
                    skip_response=False
                )
            ),
            NodeOut(
                id="order_summary",
                created=datetime.now(),
                updated=datetime.now(),
                name="Order Summary",
                is_global=False,
                global_settings=None,
                position=DisplayPosition(x=100, y=0),
                type="conversation",
                settings=ConversationSettings(
                    on_enter_text="Great! Let me summarize your order.",
                    on_enter_type="prompt",
                    allow_interruptions=True,
                    skip_response=False
                )
            )
        ],
        edges=[
            EdgeOut(
                id="edge_1",
                created=datetime.now(),
                updated=datetime.now(),
                from_node_id="greeting",
                to_node_id="order_summary",
                type="prompt",
                settings=EdgePrompt(
                    prompt="User has finished ordering, proceed to summary",
                    name="proceed_to_summary"
                )
            ),
            EdgeOut(
                id="edge_2",
                created=datetime.now(),
                updated=datetime.now(),
                from_node_id="order_summary",
                to_node_id=None,  # Terminal edge
                type="skip",
                settings=None
            )
        ]
    )
    
    # Generate the agent
    generator = CodeGenerator()
    code = generator.generate_agent(test_flow, "generated/test_pizza_agent.py")
    print("Generated agent code successfully!")
    print(f"Code length: {len(code)} characters")