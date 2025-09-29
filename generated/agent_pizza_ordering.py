import os
import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, elevenlabs, silero

# Optional provider plugins; import guarded by usage
try:
    from livekit.plugins import azure
except Exception:
    azure = None
try:
    from livekit.plugins import aws
except Exception:
    aws = None
try:
    from livekit.plugins import google
except Exception:
    google = None
from livekit import api
import aiohttp
import asyncio

# Load environment and configure logger
# Load .env then .env.local (allow .env.local to override)
load_dotenv()
load_dotenv(".env.local", override=True)
logger = logging.getLogger("pizza_ordering-agent")
# Resolve log level: explicit FACTORY_LOG_LEVEL wins; otherwise DEBUG in test mode; else INFO
_env_level = (os.getenv("FACTORY_LOG_LEVEL") or "").upper()
_level_map = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

# Generator Debug Mode (runtime toggle)
TEST_MODE = os.getenv("FACTORY_TEST_MODE", "false").lower() == "true"
# Flow generation mode (declarative default)
FLOW_GENERATION_MODE = (os.getenv("FLOW_GENERATION_MODE") or "declarative").lower()
# Apply resolved log level now that TEST_MODE is known
if _env_level in _level_map:
    logger.setLevel(_level_map[_env_level])
else:
    logger.setLevel(logging.DEBUG if TEST_MODE else logging.INFO)
if TEST_MODE:
    logger.info("FACTORY_TEST_MODE enabled: generator debug logs active")


@dataclass
class FlowState:
    """Central state management for the flow"""

    slots: Dict[str, Any] = field(default_factory=dict)
    path: List[str] = field(default_factory=list)
    task_results: Dict[str, Any] = field(default_factory=dict)
    loop_counts: Dict[str, int] = field(default_factory=dict)
    MAX_LOOP_ITERATIONS: int = field(default=10)

    def add_to_path(self, node_id: str):
        self.path.append(node_id)

        # Track loop iterations for self-loops
        if len(self.path) >= 2 and self.path[-1] == self.path[-2]:
            self.loop_counts[node_id] = self.loop_counts.get(node_id, 0) + 1
            if self.loop_counts[node_id] > self.MAX_LOOP_ITERATIONS:
                logger.warning(
                    f"Node {node_id} exceeded max loop iterations ({self.MAX_LOOP_ITERATIONS}), breaking loop"
                )
                raise Exception(f"Maximum loop iterations exceeded for node {node_id}")
        else:
            # Reset counter when leaving a node
            self.loop_counts[node_id] = 0

    def set_slot(self, key: str, value: Any):
        self.slots[key] = value

    def get_slot(self, key: str, default: Any = None) -> Any:
        return self.slots.get(key, default)


class BaseFlowAgent(Agent):
    """Base agent class with common functionality"""

    def __init__(self, job_context: JobContext, instructions: str) -> None:
        self.job_context = job_context

        # Initialize plugins based on flow settings

        stt = deepgram.STT(model="nova-2")

        llm = openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7,
        )

        tts = elevenlabs.TTS(
            api_key=(os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")),
            model="eleven_flash_v2_5",
            voice_id="21m00Tcm4TlvDq8ikWAM",
        )

        super().__init__(
            instructions=instructions, stt=stt, llm=llm, tts=tts, vad=silero.VAD.load()
        )

    def _enable_preemptive_generation(self):
        """Enable preemptive_generation once we have first user-driven tool call."""
        try:
            flow_state: FlowState = self.session.userdata
            already = flow_state.get_slot("_preemptive_enabled", False)
            if not already:
                if hasattr(self.session, "preemptive_generation"):
                    self.session.preemptive_generation = True
                flow_state.set_slot("_preemptive_enabled", True)
                logger.info("Preemptive generation enabled after first user input")
        except Exception as e:
            logger.debug("Could not toggle preemptive_generation: %s", e)

    def _route_to(self, current_node_id: str) -> Optional[Agent]:
        """Route to the next agent based on FLOW_SPEC when in declarative mode.
        Returns an Agent or None if terminal.
        """
        if FLOW_GENERATION_MODE != "declarative":
            return None  # Not used in simple mode
        try:
            spec = FLOW_SPEC.get(current_node_id)
            if not spec:
                logger.error("No FLOW_SPEC entry for %s", current_node_id)
                return None
            edges = spec.get("edges", [])
            # If single edge and it has a to_node
            if len(edges) == 1:
                if edges[0].get("to_node_id"):
                    next_node_id = edges[0]["to_node_id"]
                    if TEST_MODE:
                        logger.info(
                            "[GEN-DEBUG] router_select from=%s to=%s edge_id=%s edge_type=%s",
                            current_node_id,
                            next_node_id,
                            edges[0].get("edge_id"),
                            edges[0].get("edge_type"),
                        )
                else:
                    # Explicit terminal edge -> EndAgent
                    if TEST_MODE:
                        logger.info(
                            "[GEN-DEBUG] router_terminal from=%s to=%s (EndAgent)",
                            current_node_id,
                            None,
                        )
                    end_spec = FLOW_SPEC.get("__end__")
                    if end_spec and globals().get(end_spec.get("agent_class")):
                        return globals()[end_spec["agent_class"]](
                            job_context=self.job_context
                        )
                    return None
            else:
                # Multiple edges: do not make a naive choice here; the explicit tool determines the path
                if TEST_MODE:
                    logger.info(
                        "[GEN-DEBUG] router_multi_edges current=%s - awaiting explicit tool",
                        current_node_id,
                    )
                return None

            next_spec = FLOW_SPEC.get(next_node_id)
            if not next_spec:
                # If no next node (terminal), go to EndAgent if available
                if TEST_MODE:
                    logger.info(
                        "[GEN-DEBUG] router_terminal from=%s to=%s (EndAgent)",
                        current_node_id,
                        next_node_id,
                    )
                end_spec = FLOW_SPEC.get("__end__")
                if end_spec and globals().get(end_spec.get("agent_class")):
                    return globals()[end_spec["agent_class"]](
                        job_context=self.job_context
                    )
                return None
            agent_cls_name = next_spec.get("agent_class")
            if not agent_cls_name:
                return None
            agent_cls = globals().get(agent_cls_name)
            if not agent_cls:
                logger.error("Agent class %s not found", agent_cls_name)
                return None
            return agent_cls(job_context=self.job_context)
        except Exception as e:
            logger.error("Routing failed: %s", e)
            return None

    async def say_or_skip(self, text: Optional[str], skip_response: bool = False):
        """Say text if provided and not skipping response"""
        if text and not skip_response:
            await self.session.say(text)

    async def end_call_if_needed(self):
        """End the call and cleanup"""
        logger.info("Ending call")
        await self.session.aclose()

        try:
            request = api.DeleteRoomRequest(room=self.job_context.room.name)
            await self.job_context.api.room.delete_room(request)
            logger.info("Room deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting room: {e}")


# Declarative FLOW_SPEC (node map)
FLOW_SPEC: Dict[str, Dict[str, Any]] = {
    "greeting": {
        "agent_class": "GreetingAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_1",
                "edge_type": "prompt",
                "to_node_id": "collect_pizza_details",
                "name": "go_proceed_to_collect_pizza_details",
                "description": "Customer has provided initial order details or wants to order pizza",
            }
        ],
    },
    "collect_pizza_details": {
        "agent_class": "CollectPizzaDetailsAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_1a",
                "edge_type": "prompt",
                "to_node_id": "collect_toppings",
                "name": "go_proceed_to_toppings",
                "description": "Customer provided size/type or asked for toppings",
            }
        ],
    },
    "collect_toppings": {
        "agent_class": "CollectToppingsAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_1b",
                "edge_type": "prompt",
                "to_node_id": "ask_order_type",
                "name": "go_proceed_to_order_type",
                "description": "Customer finished selecting toppings/extras",
            }
        ],
    },
    "ask_order_type": {
        "agent_class": "AskPickupOrDeliveryAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_1c",
                "edge_type": "prompt",
                "to_node_id": "collect_address",
                "name": "go_proceed_to_address",
                "description": "Customer indicated pickup or delivery",
            },
            {
                "edge_id": "edge_1c_pickup",
                "edge_type": "prompt",
                "to_node_id": "collect_name",
                "name": "go_proceed_to_name",
                "description": "User chose pickup; skip address and proceed to name.",
            },
        ],
    },
    "collect_address": {
        "agent_class": "CollectAddressAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_1d",
                "edge_type": "prompt",
                "to_node_id": "collect_name",
                "name": "go_proceed_to_name",
                "description": "Address provided (or pickup chosen)",
            }
        ],
    },
    "collect_name": {
        "agent_class": "CollectNameAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_1e",
                "edge_type": "prompt",
                "to_node_id": "collect_order",
                "name": "go_proceed_to_phone",
                "description": "Name provided",
            }
        ],
    },
    "collect_order": {
        "agent_class": "CollectPhoneNumberAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_2",
                "edge_type": "prompt",
                "to_node_id": "send_confirmation",
                "name": "go_send_sms_confirmation",
                "description": "Customer has provided their phone number",
            }
        ],
    },
    "send_confirmation": {
        "agent_class": "SendSmsConfirmationAgent",
        "type": "function",
        "edges": [
            {
                "edge_id": "edge_3",
                "edge_type": "prompt",
                "to_node_id": "order_complete",
                "name": "go_complete_order",
                "description": "SMS confirmation has been sent successfully",
            }
        ],
    },
    "order_complete": {
        "agent_class": "OrderCompleteAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_4",
                "edge_type": "skip",
                "to_node_id": None,
                "name": "end_conversation",
                "description": "End the conversation",
            }
        ],
    },
    "__end__": {"agent_class": "EndAgent", "type": "conversation", "edges": []},
}


class EndAgent(BaseFlowAgent):
    """Terminal end node with optional FAQ handoff."""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are the end of the conversation. Offer a concise goodbye. If the user has more questions, offer to continue to FAQs.",
        )

    async def on_enter(self) -> None:
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("__end__")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "__end__",
                "conversation",
                prev_node,
            )

        # Simple goodbye; the LLM can handle follow-up intent if user speaks
        await self.say_or_skip(
            "Thank you! If you have any other questions, feel free to ask, otherwise have a great day.",
            False,
        )

    @function_tool
    async def end_conversation(self) -> Optional[Agent]:
        """Hard end the conversation"""
        await self._handle_terminal()
        return None

    @function_tool
    async def go_to_faq(self) -> Optional[Agent]:
        """Handoff to FAQ/knowledge base"""
        # Try to locate an AnswerFaqAgent class if present
        faq_cls = globals().get("AnswerFaqAgent")
        if faq_cls:
            return faq_cls(job_context=self.job_context)
        # If not present, just end
        await self._handle_terminal()
        return None

    async def _handle_terminal(self):
        # Post-call analysis is disabled for EndAgent
        await self.end_call_if_needed()


# Generated Agent Classes


class GreetingAgent(BaseFlowAgent):
    """Conversation node: greeting"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("greeting")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "greeting",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node greeting must define non-empty on_enter_text when using a prompt"

        await self.say_or_skip(
            "Hi! Welcome to Pizza Palace. What would you like to order today?", False
        )

    @function_tool
    async def go_proceed_to_collect_pizza_details(self) -> Optional[Agent]:
        """Customer has provided initial order details or wants to order pizza"""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("greeting")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "greeting",
                "conversation",
                "greeting",
                "collect_pizza_details",
                "edge_1",
                "prompt",
            )
        return CollectPizzaDetailsAgent(job_context=self.job_context)


class CollectPizzaDetailsAgent(BaseFlowAgent):
    """Conversation node: collect_pizza_details"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("collect_pizza_details")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "collect_pizza_details",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node collect_pizza_details must define non-empty on_enter_text when using a prompt"

        await self.say_or_skip(
            "Great! What size would you like (small, medium, large), and what kind of pizza?",
            False,
        )

    @function_tool
    async def go_proceed_to_toppings(self) -> Optional[Agent]:
        """Customer provided size/type or asked for toppings"""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("collect_pizza_details")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "collect_pizza_details",
                "conversation",
                "collect_pizza_details",
                "collect_toppings",
                "edge_1a",
                "prompt",
            )
        return CollectToppingsAgent(job_context=self.job_context)


class CollectToppingsAgent(BaseFlowAgent):
    """Conversation node: collect_toppings"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("collect_toppings")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "collect_toppings",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node collect_toppings must define non-empty on_enter_text when using a prompt"

        await self.say_or_skip(
            "Any toppings or extras you'd like? You can list multiple, or say 'no'.",
            False,
        )

    @function_tool
    async def go_proceed_to_order_type(self) -> Optional[Agent]:
        """Customer finished selecting toppings/extras"""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("collect_toppings")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "collect_toppings",
                "conversation",
                "collect_toppings",
                "ask_order_type",
                "edge_1b",
                "prompt",
            )
        return AskPickupOrDeliveryAgent(job_context=self.job_context)


class AskPickupOrDeliveryAgent(BaseFlowAgent):
    """Conversation node: ask_order_type"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("ask_order_type")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "ask_order_type",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node ask_order_type must define non-empty on_enter_text when using a prompt"

        await self.say_or_skip("Will this be pickup or delivery?", False)

    @function_tool
    async def go_proceed_to_address(self) -> Optional[Agent]:
        """Customer indicated pickup or delivery"""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("ask_order_type")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "ask_order_type",
                "conversation",
                "ask_order_type",
                "collect_address",
                "edge_1c",
                "prompt",
            )
        return CollectAddressAgent(job_context=self.job_context)

    @function_tool
    async def go_proceed_to_name(self) -> Optional[Agent]:
        """User chose pickup; skip address and proceed to name."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("ask_order_type")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "ask_order_type",
                "conversation",
                "ask_order_type",
                "collect_name",
                "edge_1c_pickup",
                "prompt",
            )
        return CollectNameAgent(job_context=self.job_context)


class CollectAddressAgent(BaseFlowAgent):
    """Conversation node: collect_address"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("collect_address")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "collect_address",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node collect_address must define non-empty on_enter_text when using a prompt"

        await self.say_or_skip(
            "Please provide the delivery address (street, city, zip).", False
        )

    @function_tool
    async def go_proceed_to_name(self) -> Optional[Agent]:
        """Address provided (or pickup chosen)"""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("collect_address")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "collect_address",
                "conversation",
                "collect_address",
                "collect_name",
                "edge_1d",
                "prompt",
            )
        return CollectNameAgent(job_context=self.job_context)


class CollectNameAgent(BaseFlowAgent):
    """Conversation node: collect_name"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("collect_name")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "collect_name",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node collect_name must define non-empty on_enter_text when using a prompt"

        await self.say_or_skip("What name should we put on your order?", False)

    @function_tool
    async def go_proceed_to_phone(self) -> Optional[Agent]:
        """Name provided"""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("collect_name")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "collect_name",
                "conversation",
                "collect_name",
                "collect_order",
                "edge_1e",
                "prompt",
            )
        return CollectPhoneNumberAgent(job_context=self.job_context)


class CollectPhoneNumberAgent(BaseFlowAgent):
    """Conversation node: collect_order"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("collect_order")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "collect_order",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node collect_order must define non-empty on_enter_text when using a prompt"

        await self.say_or_skip(
            "Great, lastly, what is the best phone number for your order confirmation?",
            False,
        )

    @function_tool
    async def go_send_sms_confirmation(self) -> Optional[Agent]:
        """Customer has provided their phone number"""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("collect_order")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "collect_order",
                "conversation",
                "collect_order",
                "send_confirmation",
                "edge_2",
                "prompt",
            )
        return SendSmsConfirmationAgent(job_context=self.job_context)


class SendSmsConfirmationAgent(BaseFlowAgent):
    """Function node: send_confirmation"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("send_confirmation")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "send_confirmation",
                "function",
                prev_node,
            )

        # Execute function task and handoff via session (LiveKit-aligned)
        next_agent = await self._execute_function_task()
        if next_agent:
            self.session.update_agent(next_agent)
            return None

    async def _execute_function_task(self):
        """Generic HTTP function execution"""
        flow_state: FlowState = self.session.userdata

        # Runtime validation: function node required fields
        assert (
            "https://api.example.com/sms/send" != ""
        ), "Function node send_confirmation missing required field: url"
        assert "POST" in [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
        ], "Function node send_confirmation has invalid method: POST"

        try:
            url = "https://api.example.com/sms/send"
            method = "POST"
            headers = {"Content-Type": "application/json"}

            # Interpolate body template with slots if provided

            body_template = {
                "to": "{phone}",
                "message": "Your pizza order has been confirmed! Order details: {size} {kind} pizza with {toppings}. Delivery to: {street}, {city} {zip}. Thank you!",
            }
            body = json.loads(json.dumps(body_template).format(**flow_state.slots))

            # Execute HTTP request with retries
            max_attempts = 2 + 1
            result = None

            for attempt in range(max_attempts):
                try:
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=15000 / 1000)
                    ) as session:
                        method_fn = getattr(session, method.lower())

                        if body:
                            headers.setdefault("Content-Type", "application/json")
                            async with method_fn(
                                url, json=body, headers=headers
                            ) as response:
                                response_data = (
                                    await response.json()
                                    if response.content_type == "application/json"
                                    else await response.text()
                                )
                        else:
                            async with method_fn(url, headers=headers) as response:
                                response_data = (
                                    await response.json()
                                    if response.content_type == "application/json"
                                    else await response.text()
                                )

                        result = {
                            "ok": response.status < 400,
                            "status": response.status,
                            "url": url,
                            "response": response_data,
                        }
                        break

                except Exception as e:
                    logger.error(f"HTTP request attempt {attempt + 1} failed: {e}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(1)
                    else:
                        result = {"ok": False, "error": str(e)}

            # Store result
            flow_state.task_results["send_confirmation"] = result
            logger.info(f"Function task completed: {result}")

        except Exception as e:
            logger.error(f"Function task failed: {e}")
            flow_state.task_results["send_confirmation"] = {"error": str(e)}

        # Auto-advance after function execution

        # Single edge - auto-advance

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "send_confirmation",
                "function",
                "send_confirmation",
                "order_complete",
                "edge_3",
                "prompt",
            )
        return OrderCompleteAgent(job_context=self.job_context)

    @function_tool
    async def continue_next(self) -> Optional[Agent]:
        """Continue to next node after user confirmation or function completion"""
        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("send_confirmation")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "send_confirmation",
                "function",
                "send_confirmation",
                "order_complete",
                "edge_3",
                "prompt",
            )
        return OrderCompleteAgent(job_context=self.job_context)


class OrderCompleteAgent(BaseFlowAgent):
    """Conversation node: order_complete"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("order_complete")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "order_complete",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node order_complete must define non-empty on_enter_text when using a prompt"

        await self.say_or_skip(
            "Perfect! Your pizza order has been placed and you should receive a confirmation SMS shortly. Your pizza will be ready in about 20 minutes. Thank you for choosing Pizza Palace!",
            False,
        )

    @function_tool
    async def end_conversation(self) -> Optional[Agent]:
        """End the conversation"""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("order_complete")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "order_complete",
                "conversation",
                "order_complete",
                None,
                "edge_4",
                "skip",
            )
        await self._handle_terminal()
        return None

    async def _handle_terminal(self):
        """Handle terminal node - end call"""
        await self.end_call_if_needed()


def prewarm(proc):
    """Prewarm VAD model"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the generated agent"""
    # Logging setup
    ctx.log_context_fields = {"room": ctx.room.name, "flow": "pizza_ordering"}

    # Create agent session with proper configuration
    # LLM selection

    _llm = openai.LLM(
        model="gpt-4o-mini",
        temperature=0.7,
    )

    # STT selection

    _stt = deepgram.STT(model="nova-2")

    # TTS selection

    _tts = elevenlabs.TTS(
        api_key=(os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")),
        model="eleven_flash_v2_5",
        voice_id="21m00Tcm4TlvDq8ikWAM",
    )

    session = AgentSession(
        llm=_llm,
        stt=_stt,
        tts=_tts,
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=(
            os.getenv("PREEMPTIVE_FIRST_TURN", "false").lower() == "true"
        ),
    )

    # Initialize FlowState
    session.userdata = FlowState()

    # Start with the designated start node
    start_agent = GreetingAgent(job_context=ctx)

    # Start session
    await session.start(agent=start_agent, room=ctx.room)

    # Connect to room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
