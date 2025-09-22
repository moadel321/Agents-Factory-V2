# LiveKit Agents Factory: JSON to Code Generator

## Overview

This factory is a code generation tool designed to rapidly create and deploy stateful, voice-first conversational AI agents on the LiveKit platform. It solves the problem of repeatedly building similar agent scaffolding for different business logic by adopting a "convention over configuration" approach.

You define the entire conversational flow, from initial greeting to final action, in a structured JSON file. The factory then consumes this file and outputs a single, static, and runnable Python agent file that implements your specified logic using the LiveKit Agents SDK.

This approach combines the declarative ease of a visual flow builder with the performance, reliability, and debuggability of production-grade, version-controlled code.

**Core Principles:**
- **Declarative Flow:** Define *what* the conversation should do in JSON.
- **Codegen for Performance:** The factory writes the *how* in optimized Python code.
- **Voice-First:** Built-in latency-sensitive patterns like VAD, preemptive generation, and streaming TTS.
- **Extensible:** While providing built-in tasks, the generated code is clean Python that can be easily customized.

---

## End Node and Post-Completion UX

### EndAgent (Terminal Behavior)

- The generated code includes a small `EndAgent` class and a reserved `FLOW_SPEC["__end__"]` entry.
- When the router encounters a terminal edge (`to_node_id` is `null`/`None`) or no viable next node, it falls back to `EndAgent`.
- `EndAgent`:
  - Speaks a concise goodbye on enter.
  - Exposes two tools:
    - `end_conversation()` for a hard end (runs post-call analysis and closes the room as configured).
    - `go_to_faq()` to handoff to an FAQ/KB node (if present), otherwise ends.

### Post-Completion Prompt (Schema Design)

For more natural UX, add a short post-completion conversation node after confirmations (optional in your JSON):

- `post_completion_prompt` node on-enter text: “Anything else I can help with?”
- Two edges:
  - To your FAQ/KB node (e.g., `answer_faq`).
  - To terminal (no `to_node_id`), which routes to `EndAgent`.

This keeps flows explicit and avoids abrupt endings while allowing follow-up questions.

---

## Key Concepts

The factory is built on a Directed Acyclic Graph (DAG) model with support for self-loops on conversation nodes, similar to platforms like Retell AI, but generates native LiveKit Agents code.

- **Nodes:** Represent states in the conversation. Each node becomes a dedicated `Agent` class in the generated code.
    - **Conversation Nodes:** Engage in dialogue with the user.
    - **Function Nodes:** Execute backend logic (e.g., send an SMS, transfer a call) via LiveKit `AgentTask`s.
- **Edges:** Represent the transitions between nodes. Each edge becomes a `@function_tool` that the LLM can call to move the conversation forward.
- **FlowState:** A central `dataclass` bound to `session.userdata` that holds all collected information, task results, and the conversation path.
- **BaseFlowAgent:** A generated base class that centralizes all common logic, such as plugin initialization (STT, TTS, LLM) and utilities.

---

## Getting Started: A 3-Step Guide

### Step 1: Create a Flow Definition File

First, create a JSON file (e.g., `my_pizza_flow.json`) that describes your agent. This file must adhere to the factory's schema. You can start by copying one of the provided examples in `examples/flows/`.

*For a detailed breakdown of all fields, see the **Flow JSON Schema** section below.*

### Step 2: Configure Your Environment

Create a `.env` file in the project root and add the necessary API keys and configuration.

```bash
# Required
OPENAI_API_KEY=sk-...
ELEVEN_API_KEY=...
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

# Required for STT (currently defaults to Deepgram)
DEEPGRAM_API_KEY=...

# Optional: For built-in tasks
SIP_TRUNK_ID=... # For call transfers
SMS_WEBHOOK_URL=https://... # For a generic SMS webhook
# OR
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1...

# Optional: For cleanup and formatting
DELETE_ROOM_ON_END=true
RUFF_FORMAT=1 # Set to 1 to auto-format generated code with ruff

# Optional: For testing and development
FACTORY_TEST_MODE=true # Set to true to mock all external service calls
```

### Step 3: Generate and Run the Agent

Use the provided CLI to generate the Python file from your JSON definition. Run all commands from the project root.

```bash
# 1. Generate the agent
python -m factory.cli generate --input examples/flows/pizza_flow.json --output generated/agent_pizza.py

# 2. (Optional) The generator will auto-format if RUFF_FORMAT=1 is in your .env
#    Or you can format it manually:
python -m ruff format generated/agent_pizza.py

# 3. Run the agent worker
python generated/agent_pizza.py dev
```

Your agent is now running and ready to accept connections from your LiveKit instance.

---

## Test Mode

Set `FACTORY_TEST_MODE=true` to mock SMS and call transfer tasks during development/testing:

```bash
# In .env file
FACTORY_TEST_MODE=true

# Or when running
FACTORY_TEST_MODE=true python generated/agent.py dev
```

**What it does:**
- Mocks all SMS sends and call transfers
- Returns realistic responses with generated IDs and timestamps
- Includes network delays (200ms SMS, 500ms transfers)
- Logs all operations with "TEST MODE:" prefix

**Mock responses include:** `test_mode: true`, generated message/participant IDs, and timestamps.

---

## Runtime Flags and Modes

The generator and the generated agent support a few runtime flags to control behavior and logging.

### Logging and Debugging

- `FACTORY_LOG_LEVEL` (optional): One of `CRITICAL|ERROR|WARNING|INFO|DEBUG`.
  - If set, it takes precedence for the generated agent’s logger.
- `FACTORY_TEST_MODE` (optional): When `true`:
  - Mocks external tasks (see Test Mode above), and
  - Enables generator debug logs inside the generated agent.
    - You will see concise trace lines such as:
      - `[GEN-DEBUG] enter_node node_id=<id> node_type=<type> from=<prev>`
      - `[GEN-DEBUG] transition node_id=<id> node_type=<type> from=<from> to=<to> edge_id=<edge> edge_type=<type>`

If `FACTORY_LOG_LEVEL` is not set:
- When `FACTORY_TEST_MODE=true`, the logger defaults to DEBUG.
- Otherwise, it defaults to INFO.

### Turn-Taking and Preemptive Generation

- `PREEMPTIVE_FIRST_TURN` (optional): Defaults to `false`.
  - First turn: the agent speaks only the node’s `on_enter_text` and waits for the user (no proactive LLM call).
  - After the first user-driven tool call, the agent enables `preemptive_generation` automatically for snappier follow-up turns.

### Flow Generation Mode

- `FLOW_GENERATION_MODE` (optional): `declarative` (default) or `simple`.
  - `declarative` mode:
    - A central `FLOW_SPEC` map is emitted (node_id → agent_class, type, edges).
    - A lightweight router (`_route_to`) uses `FLOW_SPEC` to compute the next Agent.
    - Conversation nodes do not call the LLM on entry; they wait for user input to drive tool selection.
  - `simple` mode:
    - Keeps explicit `go_*` tools per edge returning the next Agent directly.

---

## The Flow JSON Schema

This section details the structure of the input JSON file.

*(For the formal Pydantic models, see `factory/schema_models.py`)*

### Root Object

| Key | Type | Description |
|---|---|---|
| `id` | string | A unique identifier for the flow. |
| `url_id` | string | A URL-friendly slug used for naming the generated file. |
| `name` | string | A human-readable name for the agent. |
| `instructions` | string | The base prompt or system instructions for the LLM, applied globally. |
| `stt_settings` | object | Configuration for Speech-to-Text. See below. |
| `tts_settings` | object | Configuration for Text-to-Speech. See below. |
| `llm_settings` | object | Configuration for the Language Model. See below. |
| `call_settings` | object | Configuration for call behavior. See below. |
| `post_call_analysis`| object | (Optional) Configuration for post-call analysis. See below. |
| `start_node_id` | string | The `id` of the first node to execute when the call begins. |
| `nodes` | array | An array of Node objects. |
| `edges` | array | An array of Edge objects. |

### `stt_settings`

| Key | Type | Description |
|---|---|---|
| `provider` | string | One of `google`, `aws`, `azure`. *(Note: currently defaults to Deepgram implementation)*. |
| `language` | string | Language code, e.g., `en-US`. |

### `tts_settings` (ElevenLabs)

| Key | Type | Description |
|---|---|---|
| `tts_provider`| string | Must be `elevenlabs`. |
| `model` | string | ElevenLabs model ID, e.g., `eleven_multilingual_v2`. |
| `voice_id` | string | The ID of the voice to use. |
| `voice_settings`| object | ElevenLabs voice settings (`stability`, `similarity_boost`, etc.). |

### `llm_settings` (OpenAI)

| Key | Type | Description |
|---|---|---|
| `provider` | string | Must be `openai`. |
| `model` | string | OpenAI model ID, e.g., `gpt-4o-mini`. |
| `temperature`| number | Sampling temperature for the LLM. |
| `max_tokens`| integer | (Optional) Max output tokens. |

### `call_settings`

| Key | Type | Description |
|---|---|---|
| `who_speaks_first` | string | `agent` or `user`. |
| `end_call_on_silence_ms` | integer | Milliseconds of silence before ending the call. |
| `max_call_duration_ms`| integer | Maximum duration of the call in milliseconds. |

### Node Object

| Key | Type | Description |
|---|---|---|
| `id` | string | Unique identifier for the node. |
| `name` | string | Human-readable name. |
| `type` | string | `conversation` or `function`. |
| `settings` | object | Configuration specific to the node type. See below. |
| `function` | object | (Optional) Required if `type` is `function`. See below. |

#### `settings` (for Conversation Nodes)

| Key | Type | Description |
|---|---|---|
| `on_enter_text` | string | The text for the agent to speak upon entering this node. |
| `on_enter_type`| string | `prompt` (speak the text) or `static` (do not speak). |
| `allow_interruptions` | boolean | Whether the user can interrupt the agent's speech. |
| `skip_response` | boolean | If `true`, the agent listens but does not speak. |
| `llm_overrides`| object | (Optional) Override global `llm_settings` for this node. |

#### `function` (for Function Nodes)

| Key | Type | Description |
|---|---|---|
| `function_type`| string | One of `sms`, `call_transfer`, `rest_webhook`. |
| `parameters_schema` | object | (Optional) A JSON schema to validate and extract parameters for the task. |
| `timeout_ms` | integer | (Optional) Task execution timeout. |
| `retries` | integer | (Optional) Number of retries on failure. |

### Edge Object

| Key | Type | Description |
|---|---|---|
| `id` | string | Unique identifier for the edge. |
| `from_node_id`| string | The `id` of the source node. |
| `to_node_id` | string/null | The `id` of the destination node. `null` indicates a terminal edge that ends the call. Can be the same as `from_node_id` to create a self-loop (conversation nodes only). |
| `type` | string | `prompt` (LLM decides) or `skip` (deterministic transition). |
| `settings` | object | (Optional) Configuration for the edge. See below. |

#### `settings` (for Edges)

| Key | Type | Description |
|---|---|---|
| `prompt` | string | A description of the condition for this transition. This is shown to the LLM. |
| `name` | string | (Optional) A short, code-friendly name used to generate the `@function_tool` name. |

### Self-Loops for FAQ Patterns

The factory supports self-loops on conversation nodes, enabling FAQ-style interactions where a node can handle multiple related questions:

| Feature | Description |
|---|---|
| **Allowed on** | Conversation nodes only |
| **Prohibited on** | Function nodes (for safety) |
| **Use Case** | FAQ handling, iterative clarification, menu systems |
| **Loop Safety** | Automatic loop detection with max 10 iterations (configurable) |
| **Example** | An FAQ node that loops to itself with prompt "User has another question" |

**Important**: Multi-node cycles (e.g., A→B→A) remain prohibited. Only direct self-loops (A→A) are allowed.

**Example Edge for Self-Loop:**
```json
{
  "id": "edge_faq_loop",
  "from_node_id": "answer_faq",
  "to_node_id": "answer_faq",
  "type": "prompt",
  "settings": {
    "prompt": "User has another question - continue answering FAQs."
  }
}
```

---

## CLI Usage

The factory provides a command-line interface for common tasks.

### `generate`
Generates a single agent file from a JSON input.

```bash
python -m factory.cli generate \
  --input examples/flows/pizza.json \
  --output generated/agent_pizza.py
```

### `batch`
Generates multiple agents from a directory of JSON files.

```bash
python -m factory.cli batch \
  --input-dir examples/flows \
  --output-dir generated
```

### `validate`
Validates a flow definition file without generating code. Validation checks for DAG structure (preventing multi-node cycles) while allowing self-loops on conversation nodes for FAQ patterns.

```bash
python -m factory.cli validate --input examples/flows/pizza.json
```

---

## How it Works: Under the Hood

1.  **Parse & Validate:** The CLI reads the input JSON and parses it into Pydantic models (`factory/schema_models.py`). The validator (`factory/validator.py`) then checks for structural integrity, ensures the flow is a valid DAG (no multi-node cycles, but allows self-loops for conversation nodes), and validates settings.
2.  **Build Intermediate Representation (IR):** The valid flow object is converted into an IR (`factory/ir.py`). This step normalizes data, resolves node/edge relationships, and creates structures optimized for code generation (e.g., generating Python class and tool names).
3.  **Render Template:** The IR is passed to a Jinja2 template (`factory/templates/agent.jinja2`). This template contains the complete boilerplate for a runnable LiveKit agent, with placeholders for all the dynamic parts of your flow.
4.  **Write File:** The rendered template is written to a Python file. This file is self-contained and has no runtime dependency on the factory itself.
5.  **(Optional) Format Code:** If `RUFF_FORMAT=1` is set, the generated file is automatically formatted using `ruff`.

This process ensures that every generated agent is a clean, static, and predictable piece of code that can be versioned, tested, and deployed like any other software artifact.

---

## The Jinja2 Template (`factory/templates/agent.jinja2`)

The core of the code generation is the Jinja2 template. It defines the structure of the output Python file and uses the Intermediate Representation (IR) of your flow to fill in the details. Understanding its structure is key to customizing the factory's output.

### High-Level Structure

The `agent.jinja2` template is organized into the following sections:

1.  **Header & Imports**: Standard Python imports (`os`, `logging`, `livekit`, etc.).
2.  **FlowState Dataclass**: A static `FlowState` class is defined to manage session state.
3.  **BaseFlowAgent Class**: A base class that centralizes plugin initialization (STT, TTS, LLM) and utility methods (`say_or_skip`, `_route_to`, `end_call_if_needed`). It reads configuration directly from the `flow` object.
4.  **Task Implementations**: The Python classes for the built-in tasks (`SendSMSTask`, `TransferCallTask`, `RestWebhookTask`) are included.
5.  **Declarative FLOW_SPEC**: In `declarative` mode, the template emits a `FLOW_SPEC` (a Python dict) that maps each node to its type, class, and outgoing edges. The central router consults it for transitions, including terminal fallbacks.
6.  **Generated Agent Classes**: This is the main dynamic section. The template iterates through `flow.nodes` from the IR and generates a dedicated Python class for each node.
7.  **Entrypoint**: The standard `prewarm` and `entrypoint` functions required to run the agent as a LiveKit worker.

### How JSON Schema Maps to Template Variables

The `build_ir` function (`factory/ir.py`) transforms your JSON flow into a `flow` object that is directly accessible within the template. Here’s how the JSON properties are used:

-   **`flow.url_id`, `flow.name`, `flow.instructions`**: Used for logging, comments, and the root instructions for the `BaseFlowAgent`.
-   **`flow.llm`, `flow.tts`, `flow.stt_provider`**: These objects contain the normalized settings used to instantiate the `livekit.plugins` within `BaseFlowAgent` and the `entrypoint`.
-   **`flow.nodes` (Loop)**: The template iterates over this list. For each `node` in the list:
    -   `node.class_name`: Becomes the Python class name (e.g., `GreetingAgent`).
    -   `node.type`: Controls the logic inside the `on_enter` method. An `{% if node.type == "conversation" %}` block generates dialogue-focused code, while an `elif` handles function execution. Function nodes now perform handoff via `session.update_agent(next_agent)` immediately after the task completes (LiveKit-aligned), rather than returning an Agent from `on_enter`.
    -   `node.instructions`: Injected into the `super().__init__()` call for that specific node's agent class.
    -   `node.on_enter_text` & `node.skip_response`: Used within `on_enter` to determine if the agent should speak.
    -   `node.out_edges` (Loop): For each `edge` connected *from* the current node:
        -   `edge.tool_name`: Becomes the name of the `@function_tool` (e.g., `go_proceed_to_collect_info`).
        -   `edge.description`: Used as the docstring for the tool, which is critical for the LLM to understand its purpose.
        -   `edge.next_class_name`: Determines the return value of the tool, enabling the handoff to the next agent (e.g., `return CollectOrderDetailsAgent(...)`). If `null`, it is treated as a terminal edge; in declarative mode, the router falls back to `EndAgent`.
-   **`flow.post_call_analysis`**: If present, this object is used to generate the `_run_post_call_analysis` method, dynamically creating the prompt from the `analysis_items`.

The mapping is direct and predictable. The structure of your JSON `nodes` and `edges` arrays directly corresponds to the generated Python classes and the `@function_tool` methods that connect them.

### Creating a New Workflow Template from Scratch

While modifying the existing `agent.jinja2` is recommended, you could create a new one. Here are the principles to follow:

1.  **Start with the IR**: Your template's logic must be based on the data provided by the Intermediate Representation (IR) builder (`factory/ir.py`). If you need data that isn't there, you must first add it to the `IRFlow` dataclass and the `build_ir` function.
2.  **Core Loop is Key**: The fundamental structure is iterating through the nodes to create classes: `{% for node in flow.nodes %} ... {% endfor %}`.
3.  **Transitions are Tool-Based**: Inside the node loop, you must iterate through `node.out_edges` to generate the `@function_tool` methods that represent the valid transitions from that state.
4.  **Handle Node Types**: Use `{% if node.type == "..." %}` blocks to generate different `on_enter` logic for different kinds of nodes (e.g., speaking vs. executing a task).
5.  **Stateless Template**: The template itself should be stateless. All conversational state should be managed through the `FlowState` object and agent handoffs.
6.  **Implement LiveKit Boilerplate**: A valid template must generate all the necessary LiveKit components: a main `entrypoint`, a `prewarm` function, an `AgentSession` initialization, and the `if __name__ == "__main__":` block to make the agent runnable.
7.  **Example**: To add a new function type called `"database_query"`, you would:
    -   Update the `function_type` `Literal` in `factory/schema_models.py`.
    -   Add logic to `factory/validator.py` to validate its required parameters.
    -   Add a `DatabaseQueryTask` class to `factory/tasks.py`.
    -   In the template, add an `{% elif node.function.function_type == "database_query" %}` block inside `_execute_function_task` to instantiate and run your new task.

---

## Project Structure & File Explanations

This section provides a brief overview of the purpose of each file in the factory's source code.

### `/factory/`
This directory contains the core logic for parsing, validating, and transforming the flow JSON into a renderable structure.

-   `__init__.py`: Makes the `factory` directory a Python package.
-   `cli.py`: Defines the command-line interface (`generate`, `batch`, `validate`) using `click`.
-   `core.py`: Contains the `FlowState` dataclass and `BaseFlowAgent` used in the generated code.
-   `generator.py`: Holds the `CodeGenerator` class that orchestrates the Jinja2 template rendering.
-   `ir.py`: Converts the validated JSON schema into an Intermediate Representation for the template.
-   `prompts.py`: Helper functions for building dynamic LLM prompts for routing and analysis.
-   `schema_models.py`: Defines the Pydantic models that represent the authoritative schema for a flow JSON.
-   `tasks.py`: Contains the concrete `AgentTask` implementations for built-in functions (SMS, Call Transfer, etc.).
-   `validator.py`: Provides functions to validate the integrity, structure, and logic of a flow JSON.
-   `templates/agent.jinja2`: The Jinja2 template that defines the structure of the generated Python agent file.

### `/codegen/`
This directory contains the standalone script to run the code generation process.

-   `generate.py`: A simple script that calls the factory's core logic to generate an agent from a JSON file.
