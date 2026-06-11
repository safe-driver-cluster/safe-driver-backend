import hmac
import os
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status

from central_api.models import CommandResponse, CreateCommandRequest
from central_api.repository import FirebaseRepository

app = FastAPI(
    title="SafeDriver Central API",
    version="1.0.0",
    description="Cloud API for managing SafeDriver Raspberry Pi devices.",
)


def require_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.getenv("CENTRAL_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CENTRAL_API_KEY is not configured",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


def get_repository() -> FirebaseRepository:
    return FirebaseRepository()


Protected = Annotated[None, Depends(require_api_key)]
Repository = Annotated[FirebaseRepository, Depends(get_repository)]


@app.get("/")
def root():
    return {"name": "SafeDriver Central API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/v1/devices")
def list_devices(_: Protected, repository: Repository):
    devices = repository.list_devices()
    return [{"device_id": device_id, **data} for device_id, data in devices.items()]


@app.get("/v1/devices/{device_id}")
def get_device(device_id: str, _: Protected, repository: Repository):
    device = repository.get_device(device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"device_id": device_id, **device}


@app.post(
    "/v1/devices/{device_id}/commands",
    response_model=CommandResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_command(
    device_id: str,
    request: CreateCommandRequest,
    _: Protected,
    repository: Repository,
):
    if repository.get_device(device_id) is None:
        raise HTTPException(status_code=404, detail="Device not found")
    return repository.create_command(device_id, request.type.value, request.payload)


@app.get(
    "/v1/devices/{device_id}/commands",
    response_model=list[CommandResponse],
)
def list_commands(
    device_id: str,
    _: Protected,
    repository: Repository,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    return repository.list_commands(device_id, limit)


@app.get(
    "/v1/devices/{device_id}/commands/{command_id}",
    response_model=CommandResponse,
)
def get_command(
    device_id: str,
    command_id: str,
    _: Protected,
    repository: Repository,
):
    command = repository.get_command(device_id, command_id)
    if command is None:
        raise HTTPException(status_code=404, detail="Command not found")
    return command


@app.delete(
    "/v1/devices/{device_id}/commands/{command_id}",
    response_model=CommandResponse,
)
def cancel_command(
    device_id: str,
    command_id: str,
    _: Protected,
    repository: Repository,
):
    command = repository.get_command(device_id, command_id)
    if command is None:
        raise HTTPException(status_code=404, detail="Command not found")
    if command["status"] != "pending":
        raise HTTPException(status_code=409, detail="Only pending commands can be cancelled")
    return repository.cancel_command(device_id, command_id)
