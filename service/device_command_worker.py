import asyncio
import inspect
import logging
from datetime import datetime, timezone
from uuid import uuid4

from firebase_admin import db


logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _claim_pending_command(command_ref):
    claim_token = uuid4().hex

    def claim(command):
        if not command or command.get("status") != "pending":
            return command
        command["status"] = "processing"
        command["claim_token"] = claim_token
        command["updated_at"] = utc_now()
        return command

    command = command_ref.transaction(claim)
    if command and command.get("claim_token") == claim_token:
        return command
    return None


async def run_device_command_worker(
    device_id: str,
    handler,
    poll_interval_seconds: float = 2.0,
):
    """Poll and execute cloud commands targeted at this device."""
    commands_ref = db.reference(f"device_commands/{device_id}")
    logger.info("Device command worker started for %s", device_id)

    while True:
        try:
            commands = commands_ref.get() or {}
            pending = sorted(
                (
                    (command_id, command)
                    for command_id, command in commands.items()
                    if command.get("status") == "pending"
                ),
                key=lambda item: item[1].get("created_at", ""),
            )

            for command_id, _ in pending:
                command_ref = commands_ref.child(command_id)
                command = await asyncio.to_thread(_claim_pending_command, command_ref)
                if command is None:
                    continue

                try:
                    result = handler(command)
                    if inspect.isawaitable(result):
                        result = await result
                    update = {
                        "status": "completed",
                        "result": result,
                        "error": None,
                        "updated_at": utc_now(),
                    }
                except Exception as exc:
                    logger.exception("Device command failed: %s", command_id)
                    update = {
                        "status": "failed",
                        "result": None,
                        "error": str(exc),
                        "updated_at": utc_now(),
                    }

                await asyncio.to_thread(command_ref.update, update)
        except asyncio.CancelledError:
            logger.info("Device command worker stopped for %s", device_id)
            raise
        except Exception:
            logger.exception("Device command worker polling failed for %s", device_id)

        await asyncio.sleep(poll_interval_seconds)
