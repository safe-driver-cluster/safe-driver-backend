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


def test_unverified_movement_alert_is_disabled_when_fingerprint_is_disabled(monkeypatch):
    _drain_behavior_queue()
    monkeypatch.setattr(config, "ENABLE_FINGERPRINT", False)

    service = DriverAuthService()

    assert service.report_speed(25.0) is False
    assert _drain_behavior_queue() == []


def test_verified_driver_bypasses_auth_gate(monkeypatch):
    played = []
    monkeypatch.setattr(config, "ENABLE_FINGERPRINT", True)
    monkeypatch.setattr(
        auth_module.model_utils,
        "perform_voice_alerts",
        lambda message, label, language_dependent=True: played.append((message, label, language_dependent)),
    )

    service = DriverAuthService()
    assert service.is_verified() is False

    service.mark_verified("driver-1", "ENGLISH")

    assert service.is_verified() is True
    assert service.verified_driver() == "driver-1"
    assert played[-1][1] == config.VOICE_ALERT_FINGERPRINT_SIGNOFF_INSTRUCTION_LABEL
    assert played[-1][2] is True


def test_signed_off_driver_clears_verification(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_FINGERPRINT", True)
    monkeypatch.setattr(auth_module.model_utils, "perform_voice_alerts", lambda *args, **kwargs: None)

    service = DriverAuthService()
    service.mark_verified("driver-1", "ENGLISH")

    assert service.mark_signed_off() == "driver-1"
    assert service.is_verified() is False
    assert service.verified_driver() is None


def test_security_alert_captures_camera_frame_when_cache_is_empty(monkeypatch):
    _drain_behavior_queue()
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    class FakeCapture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            return True, frame

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(config, "ENABLE_ALERT_EVIDENCE", True)
    monkeypatch.setattr(auth_module, "get_latest_camera_frame", lambda: None)
    monkeypatch.setattr(auth_module.cv2, "VideoCapture", lambda camera_id: capture)

    service = DriverAuthService()
    assert service._emit_security_alert("security_test", "Security test", {}) is True

    event = behavior_queue.get_nowait()
    assert event["_evidence_jpeg"].startswith(b"\xff\xd8")
    assert capture.released is True


def test_linux_security_snapshot_uses_pi_safe_camera_strategy(monkeypatch):
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            return True, frame

        def release(self):
            pass

    capture = FakeCapture()
    monkeypatch.setattr(auth_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        "model.detect.create_camera_capture",
        lambda camera_id, width, height: (capture, camera_id, "rpicam-vid:0"),
    )

    opened_capture, camera_id, backend = auth_module._open_evidence_camera(0)

    assert opened_capture is capture
    assert camera_id == 0
    assert backend == "rpicam-vid:0"
