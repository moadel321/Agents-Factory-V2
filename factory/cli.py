"""
Command-line interface for the Flow Agent Factory.
"""
import os
import sys
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List
import click

from .generator import CodeGenerator, generate_from_file, generate_from_json
from .validator import validate_flow, validate_generated_code, FlowValidationError
from .schema_models import ConversationFlowOut

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_ruff_formatting(file_path: str) -> None:
    """
    Run ruff check --fix and ruff format on a Python file.

    Args:
        file_path: Path to the Python file to format
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
            timeout=30
        )
        if result.returncode != 0:
            logger.warning(f"ruff check --fix warnings: {result.stderr}")

        # Run ruff format to format code
        result = subprocess.run(
            ["ruff", "format", file_path],
            capture_output=True,
            text=True,
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


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """Flow Agent Factory - Generate LiveKit agents from flow definitions"""
    ctx.ensure_object(dict)
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")


@cli.command()
@click.option('--input', '-i', 'input_file', required=True,
              help='Input JSON file containing flow definition')
@click.option('--output', '-o', 'output_file',
              help='Output Python file (default: generated/agent_{url_id}.py)')
@click.option('--output-dir', '-d', 'output_dir', default='generated',
              help='Output directory for generated files')
@click.option('--validate/--no-validate', default=True,
              help='Validate flow before generation')
@click.option('--strict/--no-strict', default=True,
              help='Strict validation including environment checks')
@click.option('--allow-cycles', is_flag=True, default=False,
              help='Allow cycles in the flow graph (skip DAG validation)')
@click.option('--template-dir', '-t',
              help='Custom template directory')
@click.option('--stdout', is_flag=True, help='Print generated code to STDOUT instead of writing a file')
@click.option('--format', 'out_format', type=click.Choice(['text', 'json']), default='text',
              help='STDOUT format when using --stdout (default: text)')
def generate(input_file, output_file, output_dir, validate, strict, allow_cycles, template_dir, stdout, out_format):
    """Generate an agent from a flow definition file"""
    try:
        logger.info(f"Generating agent from {input_file}")
        
        # Load and parse flow definition
        with open(input_file, 'r', encoding='utf-8') as f:
            flow_data = json.load(f)
        
        flow = ConversationFlowOut(**flow_data)
        logger.info(f"Loaded flow: {flow.name} (ID: {flow.url_id})")
        
        # Validate if requested
        if validate:
            logger.info("Validating flow...")
            validate_flow(flow, strict=strict, allow_cycles=allow_cycles)
            logger.info("Flow validation passed")
        
        generator = CodeGenerator(template_dir)

        if stdout:
            # Generate but do not write to disk
            code = generator.generate_agent(flow, None, validate=False)
            # Validate generated code
            code_issues = validate_generated_code(code)
            if code_issues:
                logger.warning("Generated code validation issues:")
                for issue in code_issues:
                    logger.warning(f"  - {issue}")
            # Print to STDOUT
            if out_format == 'json':
                click.echo(json.dumps({"code": code}))
            else:
                click.echo(code)
        else:
            # Determine output path
            if not output_file:
                output_file = os.path.join(output_dir, f"agent_{flow.url_id}.py")

            code = generator.generate_agent(flow, output_file, validate=False)

            # Validate generated code
            code_issues = validate_generated_code(code)
            if code_issues:
                logger.warning("Generated code validation issues:")
                for issue in code_issues:
                    logger.warning(f"  - {issue}")

            # Format the generated code if RUFF_FORMAT is enabled
            if os.getenv('RUFF_FORMAT') == '1':
                run_ruff_formatting(output_file)

            logger.info(f"Successfully generated agent: {output_file}")
            click.echo(f"Generated agent saved to: {output_file}")
        
    except FileNotFoundError as e:
        click.echo(f"Error: Input file not found: {e}", err=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON in input file: {e}", err=True)
        sys.exit(1)
    except FlowValidationError as e:
        click.echo(f"Error: Flow validation failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during generation")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input-dir', '-i', 'input_dir', required=True,
              help='Directory containing flow JSON files')
@click.option('--output-dir', '-o', 'output_dir', default='generated',
              help='Output directory for generated agents')
@click.option('--pattern', '-p', default='*.json',
              help='File pattern to match (default: *.json)')
@click.option('--validate/--no-validate', default=True,
              help='Validate flows before generation')
@click.option('--strict/--no-strict', default=True,
              help='Strict validation including environment checks')
@click.option('--allow-cycles', is_flag=True, default=False,
              help='Allow cycles in the flow graph (skip DAG validation)')
@click.option('--template-dir', '-t',
              help='Custom template directory')
@click.option('--continue-on-error', is_flag=True,
              help='Continue processing other files if one fails')
def batch(input_dir, output_dir, pattern, validate, strict, allow_cycles, template_dir, continue_on_error):
    """Generate agents from multiple flow definition files"""
    try:
        input_path = Path(input_dir)
        if not input_path.exists():
            click.echo(f"Error: Input directory does not exist: {input_dir}", err=True)
            sys.exit(1)
        
        # Find matching files
        flow_files = list(input_path.glob(pattern))
        if not flow_files:
            click.echo(f"No files matching pattern '{pattern}' found in {input_dir}")
            return
        
        logger.info(f"Found {len(flow_files)} files to process")
        
        # Generate agents
        generator = CodeGenerator(template_dir)
        success_count = 0
        error_count = 0
        
        for flow_file in flow_files:
            try:
                logger.info(f"Processing {flow_file.name}")
                
                # Load flow
                with open(flow_file, 'r', encoding='utf-8') as f:
                    flow_data = json.load(f)
                flow = ConversationFlowOut(**flow_data)
                
                # Validate if requested
                if validate:
                    validate_flow(flow, strict=strict, allow_cycles=allow_cycles)
                
                # Generate
                output_file = os.path.join(output_dir, f"agent_{flow.url_id}.py")
                generator.generate_agent(flow, output_file, validate=False)

                # Format the generated code if RUFF_FORMAT is enabled
                if os.getenv('RUFF_FORMAT') == '1':
                    run_ruff_formatting(output_file)

                success_count += 1
                logger.info(f"✓ Generated {flow_file.name} -> {output_file}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"✗ Failed to process {flow_file.name}: {str(e)}")
                if not continue_on_error:
                    sys.exit(1)
        
        # Summary
        click.echo(f"\nBatch generation complete:")
        click.echo(f"  Success: {success_count}")
        click.echo(f"  Errors: {error_count}")
        
        if error_count > 0 and not continue_on_error:
            sys.exit(1)
            
    except Exception as e:
        logger.exception("Unexpected error during batch generation")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', 'input_file', required=True,
              help='Input JSON file containing flow definition')
@click.option('--strict/--no-strict', default=True,
              help='Strict validation including environment checks')
@click.option('--allow-cycles', is_flag=True, default=False,
              help='Allow cycles in the flow graph (skip DAG validation)')
def validate_cmd(input_file, strict, allow_cycles):
    """Validate a flow definition file"""
    try:
        logger.info(f"Validating flow from {input_file}")
        
        # Load and parse flow definition
        with open(input_file, 'r', encoding='utf-8') as f:
            flow_data = json.load(f)
        
        flow = ConversationFlowOut(**flow_data)

        # Validate
        validate_flow(flow, strict=strict, allow_cycles=allow_cycles)
        
        click.echo("✓ Flow validation passed")
        logger.info("Flow validation successful")
        
    except FileNotFoundError as e:
        click.echo(f"Error: Input file not found: {e}", err=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON in input file: {e}", err=True)
        sys.exit(1)
    except FlowValidationError as e:
        click.echo(f"✗ Flow validation failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during validation")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', 'output_file', default='example_flow.json',
              help='Output file for example flow')
def create_example(output_file):
    """Create an example flow definition file"""
    from datetime import datetime
    from .schema_models import (
        NodeOut, EdgeOut, ConversationSettings, EdgePrompt,
        STTSettings, LLMSettings, CallSettings, DisplayPosition,
        ElevenLabsTTSSettings, FunctionSettings
    )
    
    try:
        # Create example flow
        example_flow = ConversationFlowOut(
            id="example_flow_123",
            url_id="pizza_ordering",
            created=datetime.now(),
            updated=datetime.now(),
            name="Pizza Ordering Flow",
            instructions="You are a helpful pizza ordering assistant. Be friendly and efficient.",
            
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
                    name="Greeting",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=0, y=0),
                    type="conversation",
                    settings=ConversationSettings(
                        on_enter_text="Hi! Welcome to Pizza Palace. What would you like to order today?",
                        on_enter_type="prompt",
                        allow_interruptions=True,
                        skip_response=False
                    )
                ),
                
                NodeOut(
                    id="collect_order",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Collect Order",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=200, y=0),
                    type="conversation",
                    settings=ConversationSettings(
                        on_enter_text="Great! Please tell me what pizza you'd like and your phone number.",
                        on_enter_type="prompt",
                        allow_interruptions=True,
                        skip_response=False
                    )
                ),
                
                NodeOut(
                    id="send_confirmation",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Send Confirmation SMS",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=400, y=0),
                    type="function",
                    settings=FunctionSettings(
                        url="https://api.example.com/sms/send",
                        method="POST",
                        headers={"Content-Type": "application/json"},
                        body={
                            "to": "{phone}",
                            "message": "Your pizza order has been confirmed! Thank you for choosing Pizza Palace."
                        },
                        timeout_ms=15000,
                        retries=2
                    )
                ),
                
                NodeOut(
                    id="order_complete",
                    created=datetime.now(),
                    updated=datetime.now(),
                    name="Order Complete",
                    is_global=False,
                    global_settings=None,
                    position=DisplayPosition(x=600, y=0),
                    type="conversation",
                    settings=ConversationSettings(
                        on_enter_text="Perfect! Your order has been placed and you should receive a confirmation SMS shortly. Your pizza will be ready in about 20 minutes. Thank you for choosing Pizza Palace!",
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
                    to_node_id="collect_order",
                    type="prompt",
                    settings=EdgePrompt(
                        prompt="User is ready to place an order",
                        name="proceed_to_order"
                    )
                ),
                EdgeOut(
                    id="edge_2", 
                    created=datetime.now(),
                    updated=datetime.now(),
                    from_node_id="collect_order",
                    to_node_id="send_confirmation",
                    type="prompt",
                    settings=EdgePrompt(
                        prompt="User has provided their order and phone number",
                        name="send_sms_confirmation"
                    )
                ),
                EdgeOut(
                    id="edge_3",
                    created=datetime.now(),
                    updated=datetime.now(),
                    from_node_id="send_confirmation",
                    to_node_id="order_complete",
                    type="prompt",
                    settings=EdgePrompt(
                        prompt="SMS has been sent, proceed to completion",
                        name="complete_order"
                    )
                ),
                EdgeOut(
                    id="edge_4",
                    created=datetime.now(),
                    updated=datetime.now(),
                    from_node_id="order_complete", 
                    to_node_id=None,  # Terminal edge
                    type="skip",
                    settings=None
                )
            ]
        )
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(example_flow.model_dump_json(indent=2))
        
        click.echo(f"Example flow definition saved to: {output_file}")
        logger.info(f"Created example flow: {output_file}")
        
    except Exception as e:
        logger.exception("Error creating example flow")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--code-file', '-c', required=True,
              help='Generated Python code file to validate')
def validate_code(code_file):
    """Validate generated Python code"""
    try:
        with open(code_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        issues = validate_generated_code(code)
        
        if not issues:
            click.echo("✓ Generated code validation passed")
        else:
            click.echo("⚠ Generated code validation issues:")
            for issue in issues:
                click.echo(f"  - {issue}")
            sys.exit(1)
                
    except FileNotFoundError as e:
        click.echo(f"Error: Code file not found: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Error validating generated code")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()