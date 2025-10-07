import os
import logging
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


# Safe formatting helpers for templated bodies
class _SafeSlots(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _format_nested(value, mapping):
    if isinstance(value, str):
        try:
            return value.format_map(mapping)
        except Exception:
            return value
    if isinstance(value, dict):
        return {k: _format_nested(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_format_nested(v, mapping) for v in value]
    return value


# Load environment and configure logger
# Load .env then .env.local (allow .env.local to override)
load_dotenv()
load_dotenv(".env.local", override=True)
logger = logging.getLogger("dental_receptionist-agent")
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
            temperature=0.5,
        )

        tts = elevenlabs.TTS(
            api_key=(os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")),
            model="eleven_turbo_v2",
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

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """After each user turn, inject concise function results into chat context.

        This is silent (no speech); it only appends a short SYSTEM note so
        subsequent LLM generations can condition on the last function results.
        """
        try:
            flow_state: FlowState = self.session.userdata
            # Inject each result once per node id
            for _node_id, _res in (flow_state.task_results or {}).items():
                _flag_key = f"_ctx_injected_{_node_id}"
                if flow_state.slots.get(_flag_key):
                    continue
                _ok = None
                _status = None
                if isinstance(_res, dict):
                    _ok = _res.get("ok")
                    _status = _res.get("status")
                _parts = []
                if _ok is not None:
                    _parts.append(f"ok={_ok}")
                if _status is not None:
                    _parts.append(f"status={_status}")
                _summary = f"[fn:{_node_id}] " + (
                    " ".join(_parts) if _parts else "result saved"
                )
                # Append to turn context for next generation
                if hasattr(turn_ctx, "append"):
                    turn_ctx.append(text=_summary, role="system")
                    flow_state.slots[_flag_key] = True
        except Exception as e:
            logger.debug("on_user_turn_completed injection skipped: %s", e)


# Declarative FLOW_SPEC (node map)
FLOW_SPEC: Dict[str, Dict[str, Any]] = {
    "greeting_triage": {
        "agent_class": "GreetingTriageAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_1",
                "edge_type": "prompt",
                "to_node_id": "ask_patient_type",
                "name": "go_edge_1_ask_patient_type",
                "description": "User wants to book a new appointment or reschedule.",
            },
            {
                "edge_id": "edge_2",
                "edge_type": "prompt",
                "to_node_id": "verify_existing_patient_manage",
                "name": "go_Verify_Patient_Manage",
                "description": "User wants to manage, change, or cancel an existing appointment.",
            },
            {
                "edge_id": "edge_3",
                "edge_type": "prompt",
                "to_node_id": "answer_faq",
                "name": "go_Answer_FAQ",
                "description": "User is asking a general question like hours or location.",
            },
            {
                "edge_id": "edge_19",
                "edge_type": "prompt",
                "to_node_id": "execute_transfer",
                "name": "go_Execute_Call_Transfer",
                "description": "User asks to speak to a human (escalate/transfer).",
            },
        ],
    },
    "ask_patient_type": {
        "agent_class": "اسألنوعالمريضAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_4",
                "edge_type": "prompt",
                "to_node_id": "verify_existing_patient_book",
                "name": "go_Verify_Existing_Patient_Book",
                "description": "User says they are an existing patient.",
            },
            {
                "edge_id": "edge_5",
                "edge_type": "prompt",
                "to_node_id": "collect_new_patient_info",
                "name": "go_Collect_New_Patient_Info",
                "description": "User says they are a new patient.",
            },
        ],
    },
    "verify_existing_patient_book": {
        "agent_class": "VerifyExistingPatientBookAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_6",
                "edge_type": "prompt",
                "to_node_id": "offer_and_confirm_time",
                "name": "go_Offer_Confirm_Time",
                "description": "User has provided their name and date of birth.",
            }
        ],
    },
    "collect_new_patient_info": {
        "agent_class": "CollectNewPatientInfoAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_7",
                "edge_type": "prompt",
                "to_node_id": "offer_and_confirm_time",
                "name": "go_Offer_Confirm_Time",
                "description": "User has provided their name and phone number.",
            }
        ],
    },
    "offer_and_confirm_time": {
        "agent_class": "OfferConfirmTimeAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_8",
                "edge_type": "prompt",
                "to_node_id": "booking_confirmation",
                "name": "go_Booking_Confirmation",
                "description": "User has confirmed an available appointment time.",
            }
        ],
    },
    "booking_confirmation": {
        "agent_class": "BookingConfirmationAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_16",
                "edge_type": "skip",
                "to_node_id": None,
                "name": "end_conversation",
                "description": "End the conversation",
            }
        ],
    },
    "verify_existing_patient_manage": {
        "agent_class": "VerifyPatientManageAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_9",
                "edge_type": "prompt",
                "to_node_id": "offer_and_confirm_time",
                "name": "go_Offer_Confirm_Time",
                "description": "User confirms they want to reschedule their appointment.",
            },
            {
                "edge_id": "edge_10",
                "edge_type": "prompt",
                "to_node_id": "confirm_cancellation",
                "name": "go_Confirm_Cancellation",
                "description": "User confirms they want to cancel their appointment.",
            },
        ],
    },
    "confirm_cancellation": {
        "agent_class": "ConfirmCancellationAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_13",
                "edge_type": "prompt",
                "to_node_id": "goodbye",
                "name": "go_Goodbye",
                "description": "User has no more questions after cancelling.",
            }
        ],
    },
    "answer_faq": {
        "agent_class": "AnswerFaqAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_11",
                "edge_type": "prompt",
                "to_node_id": "answer_faq",
                "name": "go_Answer_FAQ",
                "description": "User has another question - continue answering FAQs.",
            },
            {
                "edge_id": "edge_12",
                "edge_type": "prompt",
                "to_node_id": "goodbye",
                "name": "go_Goodbye",
                "description": "User has no more questions.",
            },
        ],
    },
    "goodbye": {
        "agent_class": "GoodbyeAgent",
        "type": "conversation",
        "edges": [
            {
                "edge_id": "edge_15",
                "edge_type": "skip",
                "to_node_id": None,
                "name": "end_conversation",
                "description": "End the conversation",
            }
        ],
    },
    "execute_transfer": {
        "agent_class": "ExecuteCallTransferAgent",
        "type": "function",
        "edges": [
            {
                "edge_id": "edge_17",
                "edge_type": "prompt",
                "to_node_id": "demo_wait_true",
                "name": "go_Demo_Wait_True",
                "description": "Proceed to demo wait=true function",
            }
        ],
    },
    "demo_wait_true": {
        "agent_class": "DemoWaitTrueAgent",
        "type": "function",
        "edges": [
            {
                "edge_id": "edge_18",
                "edge_type": "prompt",
                "to_node_id": "goodbye",
                "name": "go_Goodbye",
                "description": "Demo done, go to goodbye",
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


class GreetingTriageAgent(BaseFlowAgent):
    """Conversation node: greeting_triage"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("greeting_triage")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "greeting_triage",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node greeting_triage (prompt) must define non-empty on_enter_text"

        await self.session.generate_reply(
            instructions="Welcome the user and say hello in a different language, be creative"
        )

    @function_tool
    async def go_edge_1_ask_patient_type(self) -> Optional[Agent]:
        """User wants to book a new appointment or reschedule."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("greeting_triage")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "greeting_triage",
                "conversation",
                "greeting_triage",
                "ask_patient_type",
                "edge_1",
                "prompt",
            )
        return اسألنوعالمريضAgent(job_context=self.job_context)

    @function_tool
    async def go_Verify_Patient_Manage(self) -> Optional[Agent]:
        """User wants to manage, change, or cancel an existing appointment."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("greeting_triage")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "greeting_triage",
                "conversation",
                "greeting_triage",
                "verify_existing_patient_manage",
                "edge_2",
                "prompt",
            )
        return VerifyPatientManageAgent(job_context=self.job_context)

    @function_tool
    async def go_Answer_FAQ(self) -> Optional[Agent]:
        """User is asking a general question like hours or location."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("greeting_triage")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "greeting_triage",
                "conversation",
                "greeting_triage",
                "answer_faq",
                "edge_3",
                "prompt",
            )
        return AnswerFaqAgent(job_context=self.job_context)

    @function_tool
    async def go_Execute_Call_Transfer(self) -> Optional[Agent]:
        """User asks to speak to a human (escalate/transfer)."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("greeting_triage")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "greeting_triage",
                "conversation",
                "greeting_triage",
                "execute_transfer",
                "edge_19",
                "prompt",
            )
        return ExecuteCallTransferAgent(job_context=self.job_context)


class اسألنوعالمريضAgent(BaseFlowAgent):
    """Conversation node: ask_patient_type"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("ask_patient_type")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "ask_patient_type",
                "conversation",
                prev_node,
            )

        await self.say_or_skip(
            "I can help with that. Are you a new or an existing patient?", False
        )

    @function_tool
    async def go_Verify_Existing_Patient_Book(self) -> Optional[Agent]:
        """User says they are an existing patient."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("ask_patient_type")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "ask_patient_type",
                "conversation",
                "ask_patient_type",
                "verify_existing_patient_book",
                "edge_4",
                "prompt",
            )
        return VerifyExistingPatientBookAgent(job_context=self.job_context)

    @function_tool
    async def go_Collect_New_Patient_Info(self) -> Optional[Agent]:
        """User says they are a new patient."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("ask_patient_type")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "ask_patient_type",
                "conversation",
                "ask_patient_type",
                "collect_new_patient_info",
                "edge_5",
                "prompt",
            )
        return CollectNewPatientInfoAgent(job_context=self.job_context)


class VerifyExistingPatientBookAgent(BaseFlowAgent):
    """Conversation node: verify_existing_patient_book"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("verify_existing_patient_book")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "verify_existing_patient_book",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node verify_existing_patient_book (prompt) must define non-empty on_enter_text"

        await self.session.generate_reply(
            instructions="Welcome back! To pull up your file, could you please tell me your full name and date of birth?"
        )

    @function_tool
    async def go_Offer_Confirm_Time(self) -> Optional[Agent]:
        """User has provided their name and date of birth."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("verify_existing_patient_book")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "verify_existing_patient_book",
                "conversation",
                "verify_existing_patient_book",
                "offer_and_confirm_time",
                "edge_6",
                "prompt",
            )
        return OfferConfirmTimeAgent(job_context=self.job_context)


class CollectNewPatientInfoAgent(BaseFlowAgent):
    """Conversation node: collect_new_patient_info"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("collect_new_patient_info")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "collect_new_patient_info",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node collect_new_patient_info (prompt) must define non-empty on_enter_text"

        await self.session.generate_reply(
            instructions="Welcome to our practice! To get you started, I'll need your full name and a good phone number to reach you at."
        )

    @function_tool
    async def go_Offer_Confirm_Time(self) -> Optional[Agent]:
        """User has provided their name and phone number."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("collect_new_patient_info")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "collect_new_patient_info",
                "conversation",
                "collect_new_patient_info",
                "offer_and_confirm_time",
                "edge_7",
                "prompt",
            )
        return OfferConfirmTimeAgent(job_context=self.job_context)


class OfferConfirmTimeAgent(BaseFlowAgent):
    """Conversation node: offer_and_confirm_time"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("offer_and_confirm_time")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "offer_and_confirm_time",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node offer_and_confirm_time (prompt) must define non-empty on_enter_text"

        await self.session.generate_reply(
            instructions="Okay, thank you. What is the reason for your visit? For example, a routine cleaning, a check-up, or are you experiencing any pain? Based on that, I can find available times."
        )

    @function_tool
    async def go_Booking_Confirmation(self) -> Optional[Agent]:
        """User has confirmed an available appointment time."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("offer_and_confirm_time")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "offer_and_confirm_time",
                "conversation",
                "offer_and_confirm_time",
                "booking_confirmation",
                "edge_8",
                "prompt",
            )
        return BookingConfirmationAgent(job_context=self.job_context)


class BookingConfirmationAgent(BaseFlowAgent):
    """Conversation node: booking_confirmation"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("booking_confirmation")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "booking_confirmation",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node booking_confirmation (prompt) must define non-empty on_enter_text"

        await self.session.generate_reply(
            instructions="Perfect. Your appointment is confirmed. You will receive a confirmation text shortly. Thank you for choosing Downtown Dental, and have a great day!"
        )

    @function_tool
    async def end_conversation(self) -> Optional[Agent]:
        """End the conversation"""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("booking_confirmation")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "booking_confirmation",
                "conversation",
                "booking_confirmation",
                None,
                "edge_16",
                "skip",
            )
        await self._handle_terminal()
        return None

    async def _handle_terminal(self):
        """Handle terminal node - end call"""
        await self.end_call_if_needed()


class VerifyPatientManageAgent(BaseFlowAgent):
    """Conversation node: verify_existing_patient_manage"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("verify_existing_patient_manage")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "verify_existing_patient_manage",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node verify_existing_patient_manage (prompt) must define non-empty on_enter_text"

        await self.session.generate_reply(
            instructions="I can certainly help with that. To find your appointment, could you please tell me your full name and date of birth?"
        )

    @function_tool
    async def go_Offer_Confirm_Time(self) -> Optional[Agent]:
        """User confirms they want to reschedule their appointment."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("verify_existing_patient_manage")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "verify_existing_patient_manage",
                "conversation",
                "verify_existing_patient_manage",
                "offer_and_confirm_time",
                "edge_9",
                "prompt",
            )
        return OfferConfirmTimeAgent(job_context=self.job_context)

    @function_tool
    async def go_Confirm_Cancellation(self) -> Optional[Agent]:
        """User confirms they want to cancel their appointment."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("verify_existing_patient_manage")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "verify_existing_patient_manage",
                "conversation",
                "verify_existing_patient_manage",
                "confirm_cancellation",
                "edge_10",
                "prompt",
            )
        return ConfirmCancellationAgent(job_context=self.job_context)


class ConfirmCancellationAgent(BaseFlowAgent):
    """Conversation node: confirm_cancellation"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("confirm_cancellation")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "confirm_cancellation",
                "conversation",
                prev_node,
            )

        assert True, "Conversation node confirm_cancellation (prompt) must define non-empty on_enter_text"

        await self.session.generate_reply(
            instructions="Alright, I have successfully canceled your appointment. Is there anything else I can help with today?"
        )

    @function_tool
    async def go_Goodbye(self) -> Optional[Agent]:
        """User has no more questions after cancelling."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("confirm_cancellation")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "confirm_cancellation",
                "conversation",
                "confirm_cancellation",
                "goodbye",
                "edge_13",
                "prompt",
            )
        return GoodbyeAgent(job_context=self.job_context)


class AnswerFaqAgent(BaseFlowAgent):
    """Conversation node: answer_faq"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("answer_faq")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "answer_faq",
                "conversation",
                prev_node,
            )

        assert (
            True
        ), "Conversation node answer_faq (prompt) must define non-empty on_enter_text"

        await self.session.generate_reply(
            instructions="Our office is open Monday to Friday from 8 AM to 5 PM. We are located at 123 Main Street. Is there anything else I can assist you with?"
        )

    @function_tool
    async def go_Answer_FAQ(self) -> Optional[Agent]:
        """User has another question - continue answering FAQs."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("answer_faq")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "answer_faq",
                "conversation",
                "answer_faq",
                "answer_faq",
                "edge_11",
                "prompt",
            )
        return AnswerFaqAgent(job_context=self.job_context)

    @function_tool
    async def go_Goodbye(self) -> Optional[Agent]:
        """User has no more questions."""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("answer_faq")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "answer_faq",
                "conversation",
                "answer_faq",
                "goodbye",
                "edge_12",
                "prompt",
            )
        return GoodbyeAgent(job_context=self.job_context)


class GoodbyeAgent(BaseFlowAgent):
    """Conversation node: goodbye"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("goodbye")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "goodbye",
                "conversation",
                prev_node,
            )

        assert (
            True
        ), "Conversation node goodbye (prompt) must define non-empty on_enter_text"

        await self.session.generate_reply(
            instructions="Thank you for calling. Have a wonderful day!"
        )

    @function_tool
    async def end_conversation(self) -> Optional[Agent]:
        """End the conversation"""
        flow_state: FlowState = self.session.userdata
        self._enable_preemptive_generation()

        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("goodbye")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "goodbye",
                "conversation",
                "goodbye",
                None,
                "edge_15",
                "skip",
            )
        await self._handle_terminal()
        return None

    async def _handle_terminal(self):
        """Handle terminal node - end call"""
        await self.end_call_if_needed()


class ExecuteCallTransferAgent(BaseFlowAgent):
    """Function node: execute_transfer"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("execute_transfer")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "execute_transfer",
                "function",
                prev_node,
            )

        # Execute function task and handoff via session (LiveKit-aligned)
        next_agent = await self._execute_function_task()
        if next_agent:
            self.session.update_agent(next_agent)
            return None

    async def _execute_function_task(self):
        """Generic HTTP function execution with optional Retell-like behavior controls"""
        flow_state: FlowState = self.session.userdata

        # Runtime validation: function node required fields
        assert (
            "https://httpbin.org/delay/2" != ""
        ), "Function node execute_transfer missing required field: url"
        assert "GET" in [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
        ], "Function node execute_transfer has invalid method: GET"

        try:
            # Optional speech during execution

            url = "https://httpbin.org/delay/2"
            method = "GET"
            headers = {}

            # Interpolate body template with slots if provided (safe recursive formatting)

            body = None

            # Helper to execute HTTP call with retries
            async def _run_http_call():
                max_attempts_local = 0 + 1
                _result = None
                for attempt in range(max_attempts_local):
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
                            _result = {
                                "ok": response.status < 400,
                                "status": response.status,
                                "url": url,
                                "response": response_data,
                            }
                            break
                    except Exception as e:
                        logger.error(f"HTTP request attempt {attempt + 1} failed: {e}")
                        if attempt < max_attempts_local - 1:
                            await asyncio.sleep(1)
                        else:
                            _result = {"ok": False, "error": str(e)}
                return _result

            # Prepare speech configuration (avoid scheduling generate_reply as a task)
            _do_speak = False
            _speak_mode = None
            _speak_text = None
            _speak_instructions = None

            _do_speak = True
            _speak_mode = "static"
            _speak_text = (
                "Testing async function call. I will move on immediately after this."
            )

            # Orchestration based on wait_for_result

            # wait_for_result = False: start HTTP in background, speak (if configured), then transition
            async def _background_call():
                _res = await _run_http_call()
                flow_state.task_results["execute_transfer"] = _res
                logger.info(f"Function task completed (background): {_res}")

            asyncio.create_task(_background_call())
            if _do_speak and _speak_mode == "static":
                await self.say_or_skip(_speak_text, False)
            elif _do_speak and _speak_mode == "prompt":
                await self.session.generate_reply(instructions=_speak_instructions)
            result = None

        except Exception as e:
            logger.error(f"Function task failed: {e}")
            flow_state.task_results["execute_transfer"] = {"error": str(e)}

        # Auto-advance after function execution (or immediately if not waiting)

        # Single edge - auto-advance

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "execute_transfer",
                "function",
                "execute_transfer",
                "demo_wait_true",
                "edge_17",
                "prompt",
            )
        return DemoWaitTrueAgent(job_context=self.job_context)

    @function_tool
    async def continue_next(self) -> Optional[Agent]:
        """Continue to next node after user confirmation or function completion"""
        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("execute_transfer")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "execute_transfer",
                "function",
                "execute_transfer",
                "demo_wait_true",
                "edge_17",
                "prompt",
            )
        return DemoWaitTrueAgent(job_context=self.job_context)


class DemoWaitTrueAgent(BaseFlowAgent):
    """Function node: demo_wait_true"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a friendly and professional dental receptionist assistant. Your goal is to handle common requests like booking appointments, managing existing ones, and answering basic questions. For any complex, urgent, or financial matters, your priority is to transfer the caller to a human receptionist.",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("demo_wait_true")
        if TEST_MODE:
            prev_node = flow_state.path[-2] if len(flow_state.path) >= 2 else None
            logger.info(
                "[GEN-DEBUG] enter_node node_id=%s node_type=%s from=%r",
                "demo_wait_true",
                "function",
                prev_node,
            )

        # Execute function task and handoff via session (LiveKit-aligned)
        next_agent = await self._execute_function_task()
        if next_agent:
            self.session.update_agent(next_agent)
            return None

    async def _execute_function_task(self):
        """Generic HTTP function execution with optional Retell-like behavior controls"""
        flow_state: FlowState = self.session.userdata

        # Runtime validation: function node required fields
        assert (
            "https://httpbin.org/delay/2" != ""
        ), "Function node demo_wait_true missing required field: url"
        assert "GET" in [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
        ], "Function node demo_wait_true has invalid method: GET"

        try:
            # Optional speech during execution

            url = "https://httpbin.org/delay/2"
            method = "GET"
            headers = {}

            # Interpolate body template with slots if provided (safe recursive formatting)

            body = None

            # Helper to execute HTTP call with retries
            async def _run_http_call():
                max_attempts_local = 0 + 1
                _result = None
                for attempt in range(max_attempts_local):
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
                            _result = {
                                "ok": response.status < 400,
                                "status": response.status,
                                "url": url,
                                "response": response_data,
                            }
                            break
                    except Exception as e:
                        logger.error(f"HTTP request attempt {attempt + 1} failed: {e}")
                        if attempt < max_attempts_local - 1:
                            await asyncio.sleep(1)
                        else:
                            _result = {"ok": False, "error": str(e)}
                return _result

            # Prepare speech configuration (avoid scheduling generate_reply as a task)
            _do_speak = False
            _speak_mode = None
            _speak_text = None
            _speak_instructions = None

            _do_speak = True
            _speak_mode = "prompt"
            _speak_instructions = (
                "Testing wait=true. I will wait for the result before moving on."
            )

            # Orchestration based on wait_for_result

            # wait_for_result = True: run HTTP concurrently, speak (if configured), then await result
            _http_task = asyncio.create_task(_run_http_call())
            if _do_speak and _speak_mode == "static":
                await self.say_or_skip(_speak_text, False)
            elif _do_speak and _speak_mode == "prompt":
                await self.session.generate_reply(instructions=_speak_instructions)
            result = await _http_task
            flow_state.task_results["demo_wait_true"] = result
            logger.info(f"Function task completed: {result}")

        except Exception as e:
            logger.error(f"Function task failed: {e}")
            flow_state.task_results["demo_wait_true"] = {"error": str(e)}

        # Auto-advance after function execution (or immediately if not waiting)

        # Single edge - auto-advance

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "demo_wait_true",
                "function",
                "demo_wait_true",
                "goodbye",
                "edge_18",
                "prompt",
            )
        return GoodbyeAgent(job_context=self.job_context)

    @function_tool
    async def continue_next(self) -> Optional[Agent]:
        """Continue to next node after user confirmation or function completion"""
        if FLOW_GENERATION_MODE == "declarative":
            next_agent = self._route_to("demo_wait_true")
            if next_agent:
                return next_agent

        if TEST_MODE:
            logger.info(
                "[GEN-DEBUG] transition node_id=%s node_type=%s from=%s to=%s edge_id=%s edge_type=%s",
                "demo_wait_true",
                "function",
                "demo_wait_true",
                "goodbye",
                "edge_18",
                "prompt",
            )
        return GoodbyeAgent(job_context=self.job_context)


def prewarm(proc):
    """Prewarm VAD model"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the generated agent"""
    # Logging setup
    ctx.log_context_fields = {"room": ctx.room.name, "flow": "dental_receptionist"}

    # Create agent session with proper configuration
    # LLM selection

    _llm = openai.LLM(
        model="gpt-4o-mini",
        temperature=0.5,
    )

    # STT selection

    _stt = deepgram.STT(model="nova-2")

    # TTS selection

    _tts = elevenlabs.TTS(
        api_key=(os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")),
        model="eleven_turbo_v2",
        voice_id="21m00Tcm4TlvDq8ikWAM",
    )

    session = AgentSession(llm=_llm, stt=_stt, tts=_tts, vad=ctx.proc.userdata["vad"])

    if hasattr(session, "preemptive_generation"):
        session.preemptive_generation = (
            os.getenv("PREEMPTIVE_FIRST_TURN", "false").lower() == "true"
        )

    # Initialize FlowState
    session.userdata = FlowState()

    # Start with the designated start node
    start_agent = GreetingTriageAgent(job_context=ctx)

    # Start session
    await session.start(agent=start_agent, room=ctx.room)

    # Connect to room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
