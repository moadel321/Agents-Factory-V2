#!/usr/bin/env python3
"""
Script to generate an example agent from the pizza flow definition.
"""
import sys
import os

# Add the factory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'factory'))

from factory.generator import generate_from_file


def main():
    """Generate the example pizza agent"""
    try:
        print("Generating pizza agent from flow definition...")
        
        input_file = "examples/pizza_flow.json"
        output_file = "generated/agent_pizza_ordering.py"
        
        # Generate the agent (with non-strict validation to skip env var checks)
        from factory.generator import CodeGenerator
        from factory.validator import validate_flow
        from factory.schema_models import ConversationFlowOut
        import json
        
        # Load flow
        with open(input_file, 'r') as f:
            flow_data = json.load(f)
        flow = ConversationFlowOut(**flow_data)
        
        # Validate with non-strict mode (skip env var checks)
        validate_flow(flow, strict=False)
        
        # Generate
        generator = CodeGenerator()
        code = generator.generate_agent(flow, output_file, validate=False)
        
        print(f"Successfully generated agent: {output_file}")
        print(f"Generated code length: {len(code)} characters")
        
        # Basic validation
        if "class FlowState" in code and "class BaseFlowAgent" in code:
            print("Generated code contains required base classes")
        else:
            print("WARNING: Generated code may be missing required classes")
            
        if "GreetingAgent" in code and "SendConfirmationAgent" in code:
            print("Generated code contains expected node agents")
        else:
            print("WARNING: Generated code may be missing expected node agents")
            
        print("\nGeneration complete! You can now run the agent with:")
        print(f"  python {output_file} console")
        
    except Exception as e:
        print(f"X Error generating agent: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()