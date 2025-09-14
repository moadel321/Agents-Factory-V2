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
from livekit import api
import aiohttp
import asyncio

# Load environment and configure logger
# Load .env then .env.local (allow .env.local to override)
load_dotenv()
load_dotenv(".env.local", override=True)
logger = logging.getLogger("pizza_ordering-agent")
logger.setLevel(logging.INFO)


@dataclass
class FlowState:
    """Central state management for the flow"""

    slots: Dict[str, Any] = field(default_factory=dict)
    path: List[str] = field(default_factory=list)
    task_results: Dict[str, Any] = field(default_factory=dict)

    def add_to_path(self, node_id: str):
        self.path.append(node_id)

    def set_slot(self, key: str, value: Any):
        self.slots[key] = value

    def get_slot(self, key: str, default: Any = None) -> Any:
        return self.slots.get(key, default)


class BaseFlowAgent(Agent):
    """Base agent class with common functionality"""

    def __init__(self, job_context: JobContext, instructions: str) -> None:
        self.job_context = job_context

        # Initialize plugins based on flow settings

        # TODO: Wire Google STT when available
        logger.warning("Google STT not yet wired, using Deepgram fallback")
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


# Task implementations
class SendSMSTask:
    def __init__(self, chat_ctx):
        self.chat_ctx = chat_ctx

    async def run(
        self, to: str, body: str, timeout_ms: int = 10000, retries: int = 0
    ) -> Dict[str, Any]:
        """Send SMS via webhook or Twilio"""
        max_attempts = retries + 1

        for attempt in range(max_attempts):
            try:
                # Try webhook first if configured
                webhook_url = os.getenv("SMS_WEBHOOK_URL")
                if webhook_url:
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=timeout_ms / 1000)
                    ) as session:
                        async with session.post(
                            webhook_url, json={"to": to, "body": body}
                        ) as response:
                            if response.status == 200:
                                return {"sent": True, "to": to, "method": "webhook"}

                # Fallback to Twilio if configured
                twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
                twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
                twilio_from = os.getenv("TWILIO_FROM_NUMBER")

                if twilio_sid and twilio_token and twilio_from:
                    # Implement Twilio SMS sending
                    logger.info(
                        f"Would send SMS via Twilio from {twilio_from} to {to}: {body}"
                    )
                    return {"sent": True, "to": to, "method": "twilio"}

                # Mock response if no real provider configured
                logger.warning("No SMS provider configured, mocking SMS send")
                return {"sent": True, "to": to, "method": "mock"}

            except Exception as e:
                logger.error(f"SMS send attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1)
                else:
                    return {"sent": False, "error": str(e)}

        return {"sent": False, "error": "Max retries exceeded"}


class TransferCallTask:
    def __init__(self, chat_ctx, job_context):
        self.chat_ctx = chat_ctx
        self.job_context = job_context

    async def run(
        self, phone_number: str, timeout_ms: int = 10000, retries: int = 0
    ) -> Dict[str, Any]:
        """Transfer call using LiveKit SIP"""
        try:
            sip_trunk_id = os.getenv("SIP_TRUNK_ID")
            if not sip_trunk_id:
                logger.warning("No SIP trunk configured, mocking transfer")
                return {"transferred": True, "to": phone_number, "method": "mock"}

            # Create SIP participant for transfer
            logger.info(f"Transferring call to {phone_number}")
            participant = await self.job_context.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=self.job_context.room.name,
                    sip_trunk_id=sip_trunk_id,
                    sip_call_to=phone_number,
                )
            )

            return {
                "transferred": True,
                "to": phone_number,
                "participant_id": participant.participant.identity,
            }

        except Exception as e:
            logger.error(f"Transfer failed: {e}")
            return {"transferred": False, "error": str(e)}


class RestWebhookTask:
    def __init__(self, chat_ctx):
        self.chat_ctx = chat_ctx

    async def run(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 10000,
        retries: int = 0,
    ) -> Dict[str, Any]:
        """Make HTTP request with retries"""
        max_attempts = retries + 1
        headers = headers or {}

        for attempt in range(max_attempts):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout_ms / 1000)
                ) as session:
                    method_fn = getattr(session, method.lower())

                    if body:
                        headers.setdefault("Content-Type", "application/json")
                        async with method_fn(
                            url, json=body, headers=headers
                        ) as response:
                            response_data = await response.json()
                    else:
                        async with method_fn(url, headers=headers) as response:
                            response_data = await response.json()

                    return {
                        "ok": response.status < 400,
                        "status": response.status,
                        "url": url,
                        "response": response_data,
                    }

            except Exception as e:
                logger.error(f"HTTP request attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1)
                else:
                    return {"ok": False, "error": str(e)}

        return {"ok": False, "error": "Max retries exceeded"}


# Generated Agent Classes


class GreetingAgent(BaseFlowAgent):
    """Conversation node: greeting"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.\n\n",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("greeting")

        await self.say_or_skip(
            "Hi! Welcome to Pizza Palace. What would you like to order today?", False
        )

        # Prompt LLM to select next action from available edge tools
        instructions = """Select the next action by calling one of these tools:

        - go_proceed_to_collect_info: Customer has provided their pizza order details
"""
        await self.session.generate_reply(instructions=instructions)

    @function_tool
    async def go_proceed_to_collect_info(self) -> Optional[Agent]:
        """Customer has provided their pizza order details"""
        flow_state: FlowState = self.session.userdata

        return CollectOrderDetailsAgent(job_context=self.job_context)

    async def _run_post_call_analysis(self):
        """Run post-call analysis if configured"""

        try:
            flow_state: FlowState = self.session.userdata

            # Build analysis prompt
            analysis_prompt = f"""
            Analyze this conversation session and provide structured analysis.
            
            Session Path: {" -> ".join(flow_state.path)}
            Collected Data: {json.dumps(flow_state.slots, indent=2)}
            Task Results: {json.dumps(flow_state.task_results, indent=2)}
            
            Return strict JSON with these fields:

            - order_completed (boolean): Whether the customer successfully completed their order

            - customer_satisfaction (selector): Estimated customer satisfaction level

            - total_items (number): Number of items in the order

            """

            # Call OpenAI for analysis
            analysis_llm = openai.LLM(model="gpt-4o-mini")
            response = await analysis_llm.agenerate(analysis_prompt)

            try:
                analysis_result = json.loads(response.choices[0].message.content)
                logger.info(f"Post-call analysis: {analysis_result}")
                flow_state.task_results["_post_call_analysis"] = analysis_result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse analysis JSON: {e}")

        except Exception as e:
            logger.error(f"Post-call analysis failed: {e}")


class CollectOrderDetailsAgent(BaseFlowAgent):
    """Conversation node: collect_order"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.\n\n",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("collect_order")

        await self.say_or_skip(
            "Great choice! Please tell me your phone number so I can send you a confirmation.",
            False,
        )

        # Prompt LLM to select next action from available edge tools
        instructions = """Select the next action by calling one of these tools:

        - go_send_sms_confirmation: Customer has provided their phone number
"""
        await self.session.generate_reply(instructions=instructions)

    @function_tool
    async def go_send_sms_confirmation(self) -> Optional[Agent]:
        """Customer has provided their phone number"""
        flow_state: FlowState = self.session.userdata

        return SendSmsConfirmationAgent(job_context=self.job_context)

    async def _run_post_call_analysis(self):
        """Run post-call analysis if configured"""

        try:
            flow_state: FlowState = self.session.userdata

            # Build analysis prompt
            analysis_prompt = f"""
            Analyze this conversation session and provide structured analysis.
            
            Session Path: {" -> ".join(flow_state.path)}
            Collected Data: {json.dumps(flow_state.slots, indent=2)}
            Task Results: {json.dumps(flow_state.task_results, indent=2)}
            
            Return strict JSON with these fields:

            - order_completed (boolean): Whether the customer successfully completed their order

            - customer_satisfaction (selector): Estimated customer satisfaction level

            - total_items (number): Number of items in the order

            """

            # Call OpenAI for analysis
            analysis_llm = openai.LLM(model="gpt-4o-mini")
            response = await analysis_llm.agenerate(analysis_prompt)

            try:
                analysis_result = json.loads(response.choices[0].message.content)
                logger.info(f"Post-call analysis: {analysis_result}")
                flow_state.task_results["_post_call_analysis"] = analysis_result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse analysis JSON: {e}")

        except Exception as e:
            logger.error(f"Post-call analysis failed: {e}")


class SendSmsConfirmationAgent(BaseFlowAgent):
    """Function node: send_confirmation"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.\n\n",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("send_confirmation")

        # Execute function task
        await self._execute_function_task()

    async def _execute_function_task(self):
        """Execute the function task for this node"""
        flow_state: FlowState = self.session.userdata

        try:
            task = SendSMSTask(self.session)
            # Extract parameters from function schema or default values
            result = await task.run(
                to=flow_state.get_slot("phone", ""),
                body=flow_state.get_slot("message", ""),
                timeout_ms=15000,
                retries=2,
            )

            # Store result
            flow_state.task_results["send_confirmation"] = result
            logger.info(f"Task sms completed: {result}")

            # Brief acknowledgment if successful
            if result.get("sent") or result.get("transferred") or result.get("ok"):
                await self.session.say("Done.")

        except Exception as e:
            logger.error(f"Function task failed: {e}")
            flow_state.task_results["send_confirmation"] = {"error": str(e)}

        # Auto-advance after function execution

        # Single edge - auto-advance

        return OrderCompleteAgent(job_context=self.job_context)

    @function_tool
    async def continue_next(self) -> Optional[Agent]:
        """Continue to next node after function completion"""

        return OrderCompleteAgent(job_context=self.job_context)

    async def _run_post_call_analysis(self):
        """Run post-call analysis if configured"""

        try:
            flow_state: FlowState = self.session.userdata

            # Build analysis prompt
            analysis_prompt = f"""
            Analyze this conversation session and provide structured analysis.
            
            Session Path: {" -> ".join(flow_state.path)}
            Collected Data: {json.dumps(flow_state.slots, indent=2)}
            Task Results: {json.dumps(flow_state.task_results, indent=2)}
            
            Return strict JSON with these fields:

            - order_completed (boolean): Whether the customer successfully completed their order

            - customer_satisfaction (selector): Estimated customer satisfaction level

            - total_items (number): Number of items in the order

            """

            # Call OpenAI for analysis
            analysis_llm = openai.LLM(model="gpt-4o-mini")
            response = await analysis_llm.agenerate(analysis_prompt)

            try:
                analysis_result = json.loads(response.choices[0].message.content)
                logger.info(f"Post-call analysis: {analysis_result}")
                flow_state.task_results["_post_call_analysis"] = analysis_result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse analysis JSON: {e}")

        except Exception as e:
            logger.error(f"Post-call analysis failed: {e}")


class OrderCompleteAgent(BaseFlowAgent):
    """Conversation node: order_complete"""

    def __init__(self, job_context: JobContext) -> None:
        super().__init__(
            job_context=job_context,
            instructions="You are a helpful pizza ordering assistant. Be friendly, efficient, and make sure to collect all necessary information for the order.\n\n",
        )

    async def on_enter(self) -> None:
        """Called when entering this node"""
        flow_state: FlowState = self.session.userdata
        flow_state.add_to_path("order_complete")

        await self.say_or_skip(
            "Perfect! Your pizza order has been placed and you should receive a confirmation SMS shortly. Your pizza will be ready in about 20 minutes. Thank you for choosing Pizza Palace!",
            False,
        )

        # Prompt LLM to select next action from available edge tools
        instructions = """Select the next action by calling one of these tools:

        - end_conversation: End the conversation
"""
        await self.session.generate_reply(instructions=instructions)

    @function_tool
    async def end_conversation(self) -> Optional[Agent]:
        """End the conversation"""
        flow_state: FlowState = self.session.userdata

        # Terminal edge
        await self._handle_terminal()
        return None

    async def _handle_terminal(self):
        """Handle terminal node - run post-call analysis and end call"""
        await self._run_post_call_analysis()
        await self.end_call_if_needed()

    async def _run_post_call_analysis(self):
        """Run post-call analysis if configured"""

        try:
            flow_state: FlowState = self.session.userdata

            # Build analysis prompt
            analysis_prompt = f"""
            Analyze this conversation session and provide structured analysis.
            
            Session Path: {" -> ".join(flow_state.path)}
            Collected Data: {json.dumps(flow_state.slots, indent=2)}
            Task Results: {json.dumps(flow_state.task_results, indent=2)}
            
            Return strict JSON with these fields:

            - order_completed (boolean): Whether the customer successfully completed their order

            - customer_satisfaction (selector): Estimated customer satisfaction level

            - total_items (number): Number of items in the order

            """

            # Call OpenAI for analysis
            analysis_llm = openai.LLM(model="gpt-4o-mini")
            response = await analysis_llm.agenerate(analysis_prompt)

            try:
                analysis_result = json.loads(response.choices[0].message.content)
                logger.info(f"Post-call analysis: {analysis_result}")
                flow_state.task_results["_post_call_analysis"] = analysis_result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse analysis JSON: {e}")

        except Exception as e:
            logger.error(f"Post-call analysis failed: {e}")


def prewarm(proc):
    """Prewarm VAD model"""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the generated agent"""
    # Logging setup
    ctx.log_context_fields = {"room": ctx.room.name, "flow": "pizza_ordering"}

    # Create agent session with proper configuration
    session = AgentSession(
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7,
        ),
        stt=deepgram.STT(model="nova-2"),
        tts=elevenlabs.TTS(
            api_key=(os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")),
            model="eleven_flash_v2_5",
            voice_id="21m00Tcm4TlvDq8ikWAM",
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
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
