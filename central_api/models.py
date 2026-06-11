from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CommandType(str, Enum):
    restart_detection = "restart_detection"
    enroll_fingerprint = "enroll_fingerprint"
    remove_fingerprint = "remove_fingerprint"
    sync_config = "sync_config"


class CreateCommandRequest(BaseModel):
    type: CommandType
    payload: dict[str, Any] = Field(default_factory=dict)


class CommandResponse(BaseModel):
    id: str
    device_id: str
    type: CommandType
    status: str
    payload: dict[str, Any]
    created_at: str
    updated_at: str
    result: Any | None = None
    error: str | None = None
