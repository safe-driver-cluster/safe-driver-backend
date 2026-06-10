import config.config as config
from gps.gps import HazardZoneMonitor, distance_meters


HAZARD = {
    "id": "hazard-1",
    "latitude": 7.235221,
    "longitude": 80.310338,
    "radius": 100.0,
}


def test_distance_meters_returns_expected_distance():
    distance = distance_meters(7.235221, 80.310338, 7.235221, 80.311238)
    assert 98.0 < distance < 102.0


def test_hazard_warns_once_until_bus_exits_buffer(monkeypatch):
    played = []
    now = [100.0]

    monkeypatch.setattr(config, "ENABLE_HAZARD_WARNINGS", True)
    monkeypatch.setattr(config, "HAZARD_FETCH_ONCE", True)
    monkeypatch.setattr(config, "HAZARD_REFRESH_INTERVAL_SEC", 60)
    monkeypatch.setattr(config, "HAZARD_EXIT_BUFFER_METERS", 20.0)
    monkeypatch.setattr(config, "LANGUAGE", "SINHALA")

    monitor = HazardZoneMonitor(
        hazard_provider=lambda: [HAZARD],
        voice_callback=lambda message, label: played.append((message, label)),
        time_provider=lambda: now[0],
    )

    # First entry warns.
    assert monitor.check_location(HAZARD["latitude"], HAZARD["longitude"]) == ["hazard-1"]
    assert len(played) == 1

    # Remaining inside and GPS jitter just outside radius do not repeat.
    assert monitor.check_location(HAZARD["latitude"], HAZARD["longitude"]) == []
    assert monitor.check_location(HAZARD["latitude"], HAZARD["longitude"] + 0.001) == []
    assert len(played) == 1

    # Exit beyond radius + buffer, then re-enter to warn again.
    monitor.check_location(HAZARD["latitude"], HAZARD["longitude"] + 0.0012)
    assert monitor.check_location(HAZARD["latitude"], HAZARD["longitude"]) == ["hazard-1"]
    assert len(played) == 2
    assert played[0][1] == config.VOICE_ALERT_HAZARD_LABEL


def test_hazard_uses_configured_alert_type(monkeypatch):
    played = []
    hazard = {
        **HAZARD,
        "type": "slippery_road",
    }

    monkeypatch.setattr(config, "ENABLE_HAZARD_WARNINGS", True)
    monkeypatch.setattr(config, "HAZARD_FETCH_ONCE", True)
    monkeypatch.setattr(config, "LANGUAGE", "ENGLISH")

    monitor = HazardZoneMonitor(
        hazard_provider=lambda: [hazard],
        voice_callback=lambda message, label: played.append((message, label)),
        time_provider=lambda: 100.0,
    )

    monitor.check_location(hazard["latitude"], hazard["longitude"])

    assert played == [
        (
            config.VOICE_ALERT_HAZARD_TYPES["slippery_road"]["messages"]["ENGLISH"],
            config.VOICE_ALERT_HAZARD_TYPES["slippery_road"]["label"],
        )
    ]


def test_hazard_type_alias_uses_configured_alert(monkeypatch):
    played = []
    hazard = {
        **HAZARD,
        "type": "accident",
    }

    monkeypatch.setattr(config, "ENABLE_HAZARD_WARNINGS", True)
    monkeypatch.setattr(config, "HAZARD_FETCH_ONCE", True)
    monkeypatch.setattr(config, "LANGUAGE", "ENGLISH")

    monitor = HazardZoneMonitor(
        hazard_provider=lambda: [hazard],
        voice_callback=lambda message, label: played.append((message, label)),
        time_provider=lambda: 100.0,
    )

    monitor.check_location(hazard["latitude"], hazard["longitude"])

    assert played == [
        (
            config.VOICE_ALERT_HAZARD_TYPES["accident_prone_zone"]["messages"]["ENGLISH"],
            config.VOICE_ALERT_HAZARD_TYPES["accident_prone_zone"]["label"],
        )
    ]


def test_unknown_hazard_type_uses_default_alert(monkeypatch):
    played = []
    hazard = {
        **HAZARD,
        "type": "unknown_hazard",
    }

    monkeypatch.setattr(config, "ENABLE_HAZARD_WARNINGS", True)
    monkeypatch.setattr(config, "HAZARD_FETCH_ONCE", True)
    monkeypatch.setattr(config, "LANGUAGE", "ENGLISH")

    monitor = HazardZoneMonitor(
        hazard_provider=lambda: [hazard],
        voice_callback=lambda message, label: played.append((message, label)),
        time_provider=lambda: 100.0,
    )

    monitor.check_location(hazard["latitude"], hazard["longitude"])

    assert played == [
        (
            config.VOICE_ALERT_HAZARD["ENGLISH"],
            config.VOICE_ALERT_HAZARD_LABEL,
        )
    ]


def test_hazard_cache_keeps_last_zones_when_refresh_fails(monkeypatch):
    now = [100.0]
    responses = iter(([HAZARD], None))

    monkeypatch.setattr(config, "ENABLE_HAZARD_WARNINGS", True)
    monkeypatch.setattr(config, "HAZARD_FETCH_ONCE", False)
    monkeypatch.setattr(config, "HAZARD_REFRESH_INTERVAL_SEC", 60)

    monitor = HazardZoneMonitor(
        hazard_provider=lambda: next(responses),
        voice_callback=lambda message, label: None,
        time_provider=lambda: now[0],
    )

    monitor.check_location(HAZARD["latitude"], HAZARD["longitude"])
    now[0] += 61
    monitor.check_location(HAZARD["latitude"], HAZARD["longitude"])

    assert monitor.hazards == [HAZARD]


def test_hazard_fetch_once_loads_firestore_only_once(monkeypatch):
    calls = []

    monkeypatch.setattr(config, "ENABLE_HAZARD_WARNINGS", True)
    monkeypatch.setattr(config, "HAZARD_FETCH_ONCE", True)

    monitor = HazardZoneMonitor(
        hazard_provider=lambda: calls.append(1) or [HAZARD],
        voice_callback=lambda message, label: None,
        time_provider=lambda: 100.0,
    )

    monitor.check_location(HAZARD["latitude"], HAZARD["longitude"])
    monitor.check_location(HAZARD["latitude"], HAZARD["longitude"])

    assert len(calls) == 1
