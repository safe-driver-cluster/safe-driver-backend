import os

from fastapi.testclient import TestClient

from central_api.app import app, get_repository


class FakeRepository:
    def __init__(self):
        self.devices = {"pi-01": {"status": "online"}}
        self.commands = {}

    def list_devices(self):
        return self.devices

    def get_device(self, device_id):
        return self.devices.get(device_id)

    def create_command(self, device_id, command_type, payload):
        command = {
            "id": "command-01",
            "device_id": device_id,
            "type": command_type,
            "status": "pending",
            "payload": payload,
            "created_at": "2026-06-12T00:00:00+00:00",
            "updated_at": "2026-06-12T00:00:00+00:00",
            "result": None,
            "error": None,
        }
        self.commands[command["id"]] = command
        return command

    def list_commands(self, device_id, limit):
        return list(self.commands.values())[:limit]

    def get_command(self, device_id, command_id):
        return self.commands.get(command_id)

    def cancel_command(self, device_id, command_id):
        self.commands[command_id]["status"] = "cancelled"
        return self.commands[command_id]


repository = FakeRepository()
app.dependency_overrides[get_repository] = lambda: repository
os.environ["CENTRAL_API_KEY"] = "test-secret"
client = TestClient(app)


def test_health_is_public():
    assert client.get("/health").json() == {"status": "healthy"}


def test_devices_require_api_key():
    assert client.get("/v1/devices").status_code == 401


def test_create_and_cancel_command():
    headers = {"X-API-Key": "test-secret"}
    response = client.post(
        "/v1/devices/pi-01/commands",
        headers=headers,
        json={"type": "restart_detection"},
    )
    assert response.status_code == 202
    assert response.json()["status"] == "pending"

    response = client.delete(
        "/v1/devices/pi-01/commands/command-01",
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
