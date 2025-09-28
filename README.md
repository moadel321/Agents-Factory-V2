# LiveKit Agents Factory: JSON to Code Generator

## Quick Start

Get up and running in minutes:

### 1. Environment Setup
```bash
# Create virtual environment (macOS/Linux)
uv venv
source .venv/bin/activate

# Create virtual environment (Windows)
uv venv
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Copy example environment file
cp .env.example .env  # macOS/Linux
copy .env.example .env  # Windows

# Edit .env file with your API keys:
# - OPENAI_API_KEY
# - DEEPGRAM_API_KEY
# - ELEVENLABS_API_KEY
# - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
```

### 3. Generate and Run Agent
```bash
# Generate agent from JSON flow
python -m factory.cli generate -i flows/pizza_flow.json

# Visualize the flow (optional)
python visualize_flow.py -i flows/pizza_flow.json

# Run the generated agent
python generated/agent_pizza_ordering.py dev
```

Your voice AI agent is now running and ready for connections!

---

## Table of Contents


- [LiveKit Agents Factory: JSON to Code Generator](#livekit-agents-factory-json-to-code-generator)
  - [Quick Start](#quick-start)
    - [1. Environment Setup](#1-environment-setup)
    - [2. Configure API Keys](#2-configure-api-keys)
    - [3. Generate and Run Agent](#3-generate-and-run-agent)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Terminal behavior and optional follow‑ups](#terminal-behavior-and-optional-followups)
  - [Key Concepts](#key-concepts)
    - [Declarative node‑level capture and tools (current design)](#declarative-nodelevel-capture-and-tools-current-design)
  - [Getting Started: A 3-Step Guide](#getting-started-a-3-step-guide)
    - [Step 1: Create a Flow Definition File](#step-1-create-a-flow-definition-file)
    - [Step 2: Configure Your Environment](#step-2-configure-your-environment)
    - [Step 3: Generate and Run the Agent](#step-3-generate-and-run-the-agent)
  - [Test Mode](#test-mode)
  - [Runtime Flags and Modes](#runtime-flags-and-modes)
    - [Logging and Debugging](#logging-and-debugging)
    - [Turn-Taking and Preemptive Generation](#turn-taking-and-preemptive-generation)
  - [The Flow JSON Schema](#the-flow-json-schema)
    - [Root Object](#root-object)
    - [`stt_settings`](#stt_settings)
    - [`tts_settings` (ElevenLabs or AWS Polly)](#tts_settings-elevenlabs-or-aws-polly)
    - [`llm_settings` (OpenAI)](#llm_settings-openai)
    - [`call_settings`](#call_settings)
    - [Node Object](#node-object)
      - [`settings` (for Conversation Nodes)](#settings-for-conversation-nodes)
      - [`function` (for Function Nodes)](#function-for-function-nodes)
    - [Edge Object](#edge-object)
      - [`settings` (for Edges)](#settings-for-edges)
    - [Self-Loops for FAQ Patterns](#self-loops-for-faq-patterns)
  - [CLI Usage](#cli-usage)
    - [`generate`](#generate)
    - [`batch`](#batch)
    - [`validate`](#validate)
  - [How it Works: Under the Hood](#how-it-works-under-the-hood)
  - [The Jinja2 Template (`factory/templates/agent.jinja2`)](#the-jinja2-template-factorytemplatesagentjinja2)
    - [High-Level Structure](#high-level-structure)
    - [How JSON Schema Maps to Template Variables](#how-json-schema-maps-to-template-variables)
  - [Project Structure \& File Explanations](#project-structure--file-explanations)
    - [`/factory/`](#factory)
  - [Schema changes performed](#schema-changes-performed)
    - [Structural differences (original-schema.json → current schema used by flows)](#structural-differences-original-schemajson--current-schema-used-by-flows)



---
## Overview

This factory is a code generation tool designed to rapidly create and deploy stateful, voice-first conversational AI agents on the LiveKit platform. It solves the problem of repeatedly building similar agent scaffolding for different business logic by adopting a "convention over configuration" approach.

You define the entire conversational flow, from initial greeting to final action, in a structured JSON file. The factory then consumes this file and outputs a single, static, and runnable Python agent file that implements your specified logic using the LiveKit Agents SDK.

This approach combines the declarative ease of a visual flow builder with the performance, reliability, and debuggability of production-grade, version-controlled code.

**Core Principles:**
- **Declarative Flow:** Define *what* the conversation should do in JSON.
- **Codegen for Performance:** The factory writes the *how* in optimized Python code.
- **Extensible:** While providing built-in tasks, the generated code is clean Python that can be easily customized.

---

## Terminal behavior and optional follow‑ups

- Terminal transitions (`to_node_id: null`) are handled by `_handle_terminal()`, which runs post‑call analysis (if configured) and ends the room.
- You may add an optional “post‑completion prompt” node (e.g., ask “Anything else I can help with?”) with two edges: one to FAQ/help and one terminal edge.

---

## Key Concepts

The factory is built on a Directed Acyclic Graph (DAG) model with support for self-loops on conversation nodes, similar to platforms like Retell AI, but generates native LiveKit Agents code.

- **Nodes:** Represent states in the conversation. Each node becomes a dedicated `Agent` class in the generated code.
    - **Conversation Nodes:** Engage in dialogue with the user.
    - **Function Nodes:** Execute backend logic (e.g., send an SMS, transfer a call) via LiveKit `AgentTask`s.
- **Edges:** Represent transitions between nodes. Each edge becomes a `@function_tool` the LLM calls to move forward.
- **FlowState:** A central `dataclass` bound to `session.userdata` that holds all collected information, task results, and the conversation path.
- **BaseFlowAgent:** A generated base class that centralizes all common logic, such as plugin initialization (STT, TTS, LLM) and utilities.

### Declarative node‑level capture and tools (current design)

- Conversation nodes with declared captures generate a single `collect(...)` tool:
  - Typed parameters (enum/string/number/boolean; lists via `multi: true`).
  - Writes values into `FlowState.slots`.
  - Auto‑advances when the node has exactly one outgoing edge; otherwise returns and awaits an explicit edge tool.
- For nodes without captures, only edge tools are generated.
- Multi‑edge routing: the central router does not pick an edge; the LLM must call the chosen edge tool.

---

## Getting Started: A 3-Step Guide

### Step 1: Create a Flow Definition File

First, create a JSON file (e.g., `my_pizza_flow.json`) that describes your agent. This file must adhere to the factory's schema. You can start by copying one of the provided examples in `flows/` (e.g., `flows/pizza_flow.json`).

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

# STT (defaults supported: Deepgram; also supports Azure and AWS)
DEEPGRAM_API_KEY=...

# Optional: Azure Speech (for Azure STT)
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=...

# Optional: AWS (for AWS Polly TTS and Amazon Transcribe STT)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Optional: Google/Gemini LLM (when llm_settings.provider=google)
GOOGLE_API_KEY=...

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
python -m factory.cli generate --input flows/pizza_flow.json --output generated/agent_pizza.py

# 2. (Optional) The generator will auto-format if RUFF_FORMAT=1 is in your .env
#    Or you can format it manually:
python -m ruff format generated/agent_pizza.py

# 3. Run the agent worker
python generated/agent_pizza.py dev
```

CLI tips:
- Print code to stdout (no file):
  - `python -m factory.cli generate -i flows/pizza_flow.json --stdout`
  - JSON-wrapped stdout: `--stdout --format json`

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
| `provider` | string | One of `google`, `aws`, `azure`, `deepgram`. |
| `language` | string | Language code, e.g., `en-US`. |
| `model` | string/null | (Optional) STT model name for providers that support models (e.g., `nova-3` for Deepgram). |

### `tts_settings` (ElevenLabs or AWS Polly)

| Key | Type | Description |
|---|---|---|
| `tts_provider`| string | `elevenlabs` or `aws`. |
| `model` | string | ElevenLabs model ID (e.g., `eleven_multilingual_v2`). For AWS, maps to the Polly engine/model (e.g., `neural`). |
| `voice_id` | string | Voice identifier (ElevenLabs voice ID or Polly voice name). |
| `voice_settings`| object | ElevenLabs voice settings (`stability`, `similarity_boost`, etc.). Optional for AWS. |

### `llm_settings` (OpenAI)

| Key | Type | Description |
|---|---|---|
| `provider` | string | One of `openai`, `azure`, `google`. |
| `model` | string | Model ID (e.g., `gpt-4.1`, `gpt-4o-mini`, `gemini-2.5-flash-lite`). |
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
| `capture` | array | (Optional) Node‑level field captures. Each item: `{ name, type: 'string'|'number'|'boolean'|'enum', enum?: string[], multi?: boolean, required?: boolean, description?: string }`. Generates a single `collect(...)` tool. |

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
3.  **Render Template:** The IR is passed to a Jinja2 template (`factory/templates/agent.jinja2`). The template emits: `FlowState`, `BaseFlowAgent`, per‑node classes, a `FLOW_SPEC`, and per‑node tools (`collect` and/or edge tools).
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
5.  **Declarative FLOW_SPEC**: Maps each node to its class/type and edges. The router auto‑advances only on single‑edge nodes.
6.  **Generated Agent Classes**: For each node, a Python class with:
   - Conversation nodes with captures: one `collect(...)` tool to record values and advance.
   - Conversation nodes without captures: only edge tools.
   - Multi‑edge nodes: no router auto‑selection; require an explicit edge tool call.
7.  **Entrypoint**: Standard `prewarm` and `entrypoint` for a LiveKit worker.

### How JSON Schema Maps to Template Variables

The `build_ir` function (`factory/ir.py`) transforms your JSON flow into a `flow` object that is directly accessible within the template. Here’s how the JSON properties are used:

-   **`flow.url_id`, `flow.name`, `flow.instructions`**: Used for logging, comments, and the root instructions for the `BaseFlowAgent`.
-   **`flow.llm`, `flow.tts`, `flow.stt`**: These objects contain the normalized settings used to instantiate the `livekit.plugins` within `BaseFlowAgent` and the `entrypoint`.
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



---

## Project Structure & File Explanations

This section provides a brief overview of the purpose of each file in the factory's source code.

### `/factory/`
This directory contains the core logic for parsing, validating, and transforming the flow JSON into a renderable structure.

-   `__init__.py`: Makes the `factory` directory a Python package.
-   `cli.py`: Defines the command-line interface (`generate`, `batch`, `validate`) using `click`.
-   `core.py`: Contains shared concepts; the generated file defines its own `FlowState` and `BaseFlowAgent`.
-   `generator.py`: Holds the `CodeGenerator` class that orchestrates the Jinja2 template rendering.
-   `ir.py`: Converts the validated JSON schema into an Intermediate Representation for the template.
-   `prompts.py`: Helper functions for building dynamic LLM prompts for routing and analysis.
-   `schema_models.py`: Defines the Pydantic models that represent the authoritative schema for a flow JSON.
-   `tasks.py`: Contains the concrete `AgentTask` implementations for built-in functions (SMS, Call Transfer, etc.).
-   `validator.py`: Provides functions to validate the integrity, structure, and logic of a flow JSON.
-   `templates/agent.jinja2`: The Jinja2 template that defines the structure of the generated Python agent file.
    - Provider-aware selection of STT (Azure/AWS/Deepgram), LLM (OpenAI/Azure/Gemini), and TTS (ElevenLabs/AWS Polly) based on `stt_settings`, `llm_settings`, and `tts_settings`.


---

## Schema changes performed

### Structural differences (original-schema.json → current schema used by flows)

- **STT providers and fields**  
  original: `STT_PROVIDERS = ["google","aws"]`; no `model` field.  
  current: `["google","aws","azure","deepgram"]` and `STTSettings.model` optional (e.g., Deepgram `nova-3`).

- **TTS (ElevenLabs/AWS)**  
  original: ElevenLabs `model: str` (free-form); AWS Polly supported.  
  current: ElevenLabs `model` is a strict Literal set (`eleven_multilingual_v2`, `eleven_flash_v2_5`, etc.); AWS Polly still supported. `voice_settings` fields are optional (`style`/`speed`/`use_speaker_boost`) instead of `Union[..., None]`.

- **LLM settings**  
  original: `model` Literal constrained to `['gpt-4.1','azure-gpt-4.1']`.  
  current: `model: str` (unconstrained to support OpenAI, Azure OpenAI, Gemini, etc.); `provider` unchanged (`openai|azure|google`).

- **Post‑call analysis**  
  original: `model` limited to `gpt-4.1`, `gpt-4.1-mini`.  
  current: expanded to `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`; `analysis_items` unchanged semantically.

- **Conversation settings**  
  original: no `llm_overrides`, no structured captures.  
  current: adds `llm_overrides` (per-node LLM tweaks) and `capture` fields (typed, enum, multi, required, description).

- **Function nodes support**  
  original: `NodeOut` lacks a `function` configuration; function behavior not modeled.  
  current: adds `FunctionSettings` and `NodeOut.function` (supports `sms`, `call_transfer`, `rest_webhook` + JSON schema, timeout, retries).

- **Edges**  
  original: `EdgePrompt` has only `prompt`.  
  current: adds optional `name` to `EdgePrompt` (used to generate stable tool names).

- **Flow**  
  both: include `call_settings`, `begin_position`, `start_node_id` (optional), `nodes`, `edges`.  
  current validator behavior (not schema fields): enforces DAG, allows conversation self‑loops, forbids multi‑node cycles.

