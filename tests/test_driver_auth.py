import numpy as np

import config.config as config
import service.driver_auth_service as auth_module
from service.driver_auth_service import DriverAuthService
from shared import behavior_queue


def _drain_behavior_queue():
    items = []
    while not behavior_queue.empty():
        items.append(behavior_queue.get_nowait())
    return items


def test_unauthorized_attempts_warn_three_times_then_cloud_alert(monkeypatch):
    _drain_behavior_queue()
    voices = []

    monkeypatch.setattr(config, "ENABLE_FINGERPRINT", True)
    monkeypatch.setattr(config, "ENABLE_ALERT_EVIDENCE", True)
    monkeypatch.setattr(config, "UNAUTHORIZED_FINGERPRINT_WARNING_LIMIT", 3)
    monkeypatch.setattr(
        auth_module.model_utils,
        "perform_voice_alerts",
        lambda message, label: voices.append((message, label)),
    )
    monkeypatch.setattr(
        auth_module,
        "get_latest_camera_frame",
        lambda: np.zeros((24, 32, 3), dtype=np.uint8),
    )

    service = DriverAuthService()
    for _ in range(4):
        service.record_unauthorized_attempt()

    events = _drain_behavior_queue()
    assert len(voices) == 3
    assert len(events) == 1
    assert events[0]["type"] == config.BEHAVIOR_UNAUTHORIZED_DRIVER
    assert events[0]["_evidence_jpeg"].startswith(b"\xff\xd8")


def test_unverified_movement_alerts_once_until_bus_stops(monkeypatch):
    _drain_behavior_queue()
    now = [1000.0]

    monkeypatch.setattr(config, "ENABLE_FINGERPRINT", True)
    monkeypatch.setattr(config, "ENABLE_ALERT_EVIDENCE", True)
    monkeypatch.setattr(config, "DETECTION_ENABLE_SPEED_KMPH", 20.0)
    monkeypatch.setattr(config, "UNVERIFIED_MOVEMENT_ALERT_COOLDOWN_SEC", 300.0)
    monkeypatch.setattr(auth_module.time, "time", lambda: now[0])
    monkeypatch.setattr(
        auth_module,
        "get_latest_camera_frame",
        lambda: np.zeros((24, 32, 3), dtype=np.uint8),
    )

    service = DriverAuthService()
    assert service.report_speed(25.0) is True
    assert service.report_speed(30.0) is False
    assert service.report_speed(0.0) is False
    assert service.report_speed(25.0) is True

    events = _drain_behavior_queue()
    assert [event["type"] for event in events] == [
        config.BEHAVIOR_UNVERIFIED_DRIVER_MOVEMENT,
        config.BEHAVIOR_UNVERIFIED_DRIVER_MOVEMENT,
    ]


def test_verified_driver_bypasses_auth_gate(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_FINGERPRINT", True)
    monkeypatch.setattr(auth_module.model_utils, "perform_voice_alerts", lambda *args: None)

    service = DriverAuthService()
    assert service.is_verified() is False

    service.mark_verified("driver-1", "ENGLISH")

    assert service.is_verified() is True
    assert service.verified_driver() == "driver-1"
