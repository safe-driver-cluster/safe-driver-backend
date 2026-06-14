import io
import logging

import numpy as np

import config.config as config
import model.alerts as alerts_module
from model.alerts import AlertManager
from shared import behavior_queue


def _drain_behavior_queue():
    items = []
    while not behavior_queue.empty():
        items.append(behavior_queue.get_nowait())
    return items


def _send_microsleep(manager, cycle_id):
    manager.check_and_send_threshold_alert(
        tag="DROWSY_EVENT",
        event_type=config.BEHAVIOR_MICROSLEEP,
        message="Microsleep detected",
        behavior_data={"duration": 1.5},
        policy_key="microsleep",
        cycle_id=cycle_id,
        trigger_voice=True,
        trigger_buzzer=True,
    )


def test_behavior_alert_does_not_use_hardcoded_driver():
    _drain_behavior_queue()
    manager = AlertManager(
        logger=logging.getLogger("test-alerts"),
        now_provider=lambda: "now",
        output_stream=io.StringIO(),
    )

    manager.send_behavior_to_parent(
        event_type="test",
        message="test",
        behavior_data={"value": 1},
    )

    payload = behavior_queue.get_nowait()
    assert payload["driver"] == ""
    assert "driver_id" not in payload


def test_microsleep_emits_three_spaced_voice_levels(monkeypatch):
    now = [0.0]
    played = []
    buzzed = []

    monkeypatch.setattr(alerts_module.time, "time", lambda: now[0])
    monkeypatch.setattr(config, "VOICE_ALERT_CONSECUTIVE_EVENT_THRESH", 3)
    monkeypatch.setattr(config, "BUZZER_ALERT_CONSECUTIVE_EVENT_THRESH", 3)
    monkeypatch.setattr(config, "MAXIMUM_BUZZER_ALERTS_PER_TYPE", 2)
    monkeypatch.setattr(config, "MAXIMUM_VOICE_ALERTS_PER_TYPE", 3)
    monkeypatch.setattr(config, "BUZZER_ALERT_COOLDOWN_SEC", 5.0)
    monkeypatch.setattr(config, "VOICE_ALERT_COOLDOWN_SEC", 10.0)
    monkeypatch.setattr(
        alerts_module.utils,
        "perform_voice_alerts",
        lambda message, label: played.append(label),
    )

    manager = AlertManager(
        logger=logging.getLogger("test-alerts"),
        now_provider=lambda: "now",
        output_stream=io.StringIO(),
        buzzer_callback=lambda: buzzed.append(now[0]),
    )

    # Production sequence: two initial events build the consecutive count,
    # events 3-4 buzzer, then later events emit voice levels 1-3.
    for index, timestamp in enumerate((0.0, 6.0, 12.0, 25.0, 31.0, 68.0, 79.0), start=1):
        now[0] = timestamp
        _send_microsleep(manager, cycle_id=index)

    assert buzzed == [12.0, 25.0]
    assert played == [
        "VOICE_ALERT_MICROSLEEP",
        "VOICE_ALERT_MICROSLEEP_L2",
        "VOICE_ALERT_MICROSLEEP_L3",
    ]


def test_microsleep_voice_level_waits_for_cooldown(monkeypatch):
    now = [0.0]
    played = []

    monkeypatch.setattr(alerts_module.time, "time", lambda: now[0])
    monkeypatch.setattr(config, "VOICE_ALERT_CONSECUTIVE_EVENT_THRESH", 1)
    monkeypatch.setattr(config, "BUZZER_ALERT_CONSECUTIVE_EVENT_THRESH", 1)
    monkeypatch.setattr(config, "MAXIMUM_BUZZER_ALERTS_PER_TYPE", 0)
    monkeypatch.setattr(config, "MAXIMUM_VOICE_ALERTS_PER_TYPE", 3)
    monkeypatch.setattr(config, "VOICE_ALERT_COOLDOWN_SEC", 10.0)
    monkeypatch.setattr(
        alerts_module.utils,
        "perform_voice_alerts",
        lambda message, label: played.append(label),
    )

    manager = AlertManager(
        logger=logging.getLogger("test-alerts"),
        now_provider=lambda: "now",
        output_stream=io.StringIO(),
    )

    for index, timestamp in enumerate((0.0, 5.0, 10.0, 20.0), start=1):
        now[0] = timestamp
        _send_microsleep(manager, cycle_id=index)

    assert played == [
        "VOICE_ALERT_MICROSLEEP",
        "VOICE_ALERT_MICROSLEEP_L2",
        "VOICE_ALERT_MICROSLEEP_L3",
    ]


def test_object_voice_alert_uses_sinhala_text_with_base_label(monkeypatch):
    monkeypatch.setattr(config, "LANGUAGE", "SINHALA")

    manager = AlertManager(
        logger=logging.getLogger("test-alerts"),
        now_provider=lambda: "now",
        output_stream=io.StringIO(),
    )

    voice_text, voice_label = manager.get_voice_msg_by_level(
        config.BEHAVIOR_MOBILE_USE,
        level=1,
    )

    assert voice_text == config.VOICE_ALERT_PHONE_SINHALA
    assert voice_label == "VOICE_ALERT_PHONE"


def test_cloud_alerts_continue_until_timeframe_limit(monkeypatch):
    _drain_behavior_queue()
    monkeypatch.setattr(config, "MICROSLEEP_EVENT_COUNT_THRESH", 7)
    monkeypatch.setattr(config, "CLOUD_ALERT_TIMEFRAME_COUNT_LIMIT", 10)

    manager = AlertManager(
        logger=logging.getLogger("test-alerts"),
        now_provider=lambda: "now",
        output_stream=io.StringIO(),
    )

    for count in range(1, 12):
        manager.check_and_send_threshold_alert(
            tag="DROWSY_EVENT",
            event_type=config.BEHAVIOR_MICROSLEEP,
            message="Microsleep detected",
            behavior_data={"duration": 1.5},
            policy_key="microsleep",
            current_count=count,
            threshold=config.MICROSLEEP_EVENT_COUNT_THRESH,
            send_cloud=True,
            timeframe_count=count,
        )

    events = _drain_behavior_queue()

    assert [event["data"]["timeframe_count"] for event in events] == [7, 8, 9, 10]


def test_cloud_alert_contains_encoded_evidence_frame(monkeypatch):
    _drain_behavior_queue()
    monkeypatch.setattr(config, "ENABLE_ALERT_EVIDENCE", True)
    monkeypatch.setattr(config, "ALERT_EVIDENCE_JPEG_QUALITY", 80)

    manager = AlertManager(
        logger=logging.getLogger("test-alerts"),
        now_provider=lambda: "now",
        output_stream=io.StringIO(),
    )
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    manager.check_and_send_threshold_alert(
        tag="DROWSY_EVENT",
        event_type=config.BEHAVIOR_MICROSLEEP,
        message="Microsleep detected",
        behavior_data={"duration": 1.5},
        send_cloud=True,
        evidence_frame=frame,
    )

    event = _drain_behavior_queue()[0]
    assert event["_evidence_jpeg"].startswith(b"\xff\xd8")
    assert "_evidence_jpeg" not in event["data"]
