from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from typing import Any

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    RoomInputOptions,
    WorkerOptions,
)
from livekit.plugins import (
    deepgram,
    openai,
    cartesia,
    silero,
    turn_detector,
)


# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class OutboundCaller(Agent):
    def __init__(
        self,
        *,
        name: str,
        appointment_time: str,
        dial_info: dict[str, Any],
    ):
        super().__init__(
            instructions=f"""
            You're a sales rep for Futurense Technologies. You're calling {name} about a Data Science certification program at IIT Mandi.
            
            Follow this conversation structure:
            1) Introduce yourself by saying "Hi, I'm calling from Futurense Technologies. Do you have a minute to talk?" (STOP)
            2) After they respond, say "I see that you're interested in our Data Science certification program at IIT Mandi" (STOP)
            3) After they respond, ask "Where did you study from?" (STOP)
            4) After they respond, ask about their graduation percentage/CGPA (STOP)
            5) Thank them and say someone will reach out, then end the call
            
            Additional guidelines:
            - Speak one dialogue at a time. Speak less and listen more.
            - Talk at a slightly fast pace like a typical sales call.
            - If the user expresses disinterest or negativity, politely apologize and end the call.
            - If the user wants to be transferred to a human agent, use the transfer_call tool.
            
            Your interface with the user will be voice. Be conversational but concise.
            """
        )
        # keep reference to the participant for transfers
        self.participant: rtc.RemoteParticipant | None = None

        self.dial_info = dial_info

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Transfer the call to a human agent, called after confirming with the user"""

        transfer_to = self.dial_info["transfer_to"]
        if not transfer_to:
            return "cannot transfer call"

        logger.info(f"transferring call to {transfer_to}")

        # let the message play fully before transferring
        await ctx.session.generate_reply(
            instructions="let the user know you'll be transferring them"
        )

        job_ctx = get_job_context()
        try:
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )

            logger.info(f"transferred call to {transfer_to}")
        except Exception as e:
            logger.error(f"error transferring call: {e}")
            await ctx.session.generate_reply(
                instructions="there was an error transferring the call."
            )
            await self.hangup()

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        logger.info(f"ending the call for {self.participant.identity}")

        # let the agent finish speaking
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.done()

        await self.hangup()

    @function_tool()
    async def look_up_availability(
        self,
        ctx: RunContext,
        date: str,
    ):
        """Called when the user asks about alternative appointment availability

        Args:
            date: The date of the appointment to check availability for
        """
        logger.info(
            f"looking up availability for {self.participant.identity} on {date}"
        )
        await asyncio.sleep(3)
        return {
            "available_times": ["1pm", "2pm", "3pm"],
        }

    @function_tool()
    async def confirm_appointment(
        self,
        ctx: RunContext,
        date: str,
        time: str,
    ):
        """Called when the user confirms their appointment on a specific date.
        Use this tool only when they are certain about the date and time.

        Args:
            date: The date of the appointment
            time: The time of the appointment
        """
        logger.info(
            f"confirming appointment for {self.participant.identity} on {date} at {time}"
        )
        return "reservation confirmed"

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        logger.info(f"detected answering machine for {self.participant.identity}")
        await self.hangup()


async def entrypoint(ctx: JobContext):
    global _default_instructions, outbound_trunk_id
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # when dispatching the agent, we'll pass it the approriate info to dial the user
    # dial_info is a dict with the following keys:
    # - phone_number: the phone number to dial
    # - transfer_to: the phone number to transfer the call to when requested
    try:
        # Try to parse JSON or handle various formats
        if ctx.job.metadata:
            logger.info(f"Raw metadata: {ctx.job.metadata}")
            # Check if the metadata is already a string representation of a dict
            if isinstance(ctx.job.metadata, str):
                # Try to parse as JSON first
                try:
                    dial_info = json.loads(ctx.job.metadata)
                except json.JSONDecodeError:
                    # It could be a string that's quoted/escaped incorrectly
                    # Try to extract the phone number using a simple match
                    import re
                    phone_match = re.search(r'phone_number[\'"\s:]+([+\d]+)', ctx.job.metadata)
                    transfer_match = re.search(r'transfer_to[\'"\s:]+([+\d]+)', ctx.job.metadata)
                    
                    dial_info = {}
                    if phone_match:
                        dial_info["phone_number"] = phone_match.group(1)
                    if transfer_match:
                        dial_info["transfer_to"] = transfer_match.group(1)
                    
                    logger.info(f"Extracted phone info using regex: {dial_info}")
            else:
                # If it's already a dict, use it directly
                dial_info = ctx.job.metadata
        else:
            dial_info = {}
    except Exception as e:
        logger.warning(f"Error parsing metadata: {e}. Using default values")
        dial_info = {}
    
    # Set default values if not provided in metadata
    if "phone_number" not in dial_info:
        dial_info["phone_number"] = os.getenv("DEFAULT_PHONE_NUMBER", "")
    if "transfer_to" not in dial_info:
        dial_info["transfer_to"] = os.getenv("DEFAULT_TRANSFER_NUMBER", "")
    if "prospect_name" not in dial_info:
        dial_info["prospect_name"] = "there"  # Default greeting if no name is provided

    logger.info(f"Using dial_info: {dial_info}")

    # look up the user's phone number and appointment details
    agent = OutboundCaller(
        name=dial_info.get("prospect_name", "there"),
        appointment_time="next Tuesday at 3pm",  # This is not used in the sales script but required by the constructor
        dial_info=dial_info,
    )

    # the following uses GPT-4o, Deepgram and Cartesia
    session = AgentSession(
        turn_detection="vad",  # Use simpler VAD-based turn detection
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        # you can also use OpenAI's TTS with openai.TTS()
        tts=cartesia.TTS(),
        llm=openai.LLM(model="gpt-4o"),
        # you can also use a speech-to-speech model like OpenAI's Realtime API
        # llm=openai.realtime.RealtimeModel()
    )

    # start the session first before dialing, to ensure that when the user picks up
    # the agent does not miss anything the user says
    # creating a task for this because session.start does not return until the participant is available
    asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
        )
    )

    # `create_sip_participant` starts dialing the user
    try:
        # Only proceed with the call if we have a phone number
        if not dial_info["phone_number"]:
            logger.error("Cannot make outbound call: no phone number provided")
            ctx.shutdown()
            return
            
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=dial_info["phone_number"],
                participant_identity="phone_user",
                # function blocks until user answers the call, or if the call fails
                wait_until_answered=True,
            )
        )

        # a participant phone user is now available
        participant = await ctx.wait_for_participant(identity="phone_user")
        agent.set_participant(participant)

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )
