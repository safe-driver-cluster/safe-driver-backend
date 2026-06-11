from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from central_api.firebase import reference


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class FirebaseRepository:
    def list_devices(self) -> dict[str, Any]:
        return reference("devices").get() or {}

    def get_device(self, device_id: str) -> dict[str, Any] | None:
        return reference(f"devices/{device_id}").get()

    def create_command(
        self,
        device_id: str,
        command_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        command_id = uuid4().hex
        timestamp = utc_now()
        command = {
            "id": command_id,
            "device_id": device_id,
            "type": command_type,
            "status": "pending",
            "payload": payload,
            "created_at": timestamp,
            "updated_at": timestamp,
            "result": None,
            "error": None,
        }
        reference(f"device_commands/{device_id}/{command_id}").set(command)
        return command

    def list_commands(self, device_id: str, limit: int) -> list[dict[str, Any]]:
        commands = (
            reference(f"device_commands/{device_id}")
            .order_by_child("created_at")
            .limit_to_last(limit)
            .get()
            or {}
        )
        return sorted(commands.values(), key=lambda item: item["created_at"], reverse=True)

    def get_command(self, device_id: str, command_id: str) -> dict[str, Any] | None:
        return reference(f"device_commands/{device_id}/{command_id}").get()

    def cancel_command(self, device_id: str, command_id: str) -> dict[str, Any] | None:
        command_ref = reference(f"device_commands/{device_id}/{command_id}")

        def cancel_if_pending(command):
            if command and command.get("status") == "pending":
                command["status"] = "cancelled"
                command["updated_at"] = utc_now()
            return command

        return command_ref.transaction(cancel_if_pending)
