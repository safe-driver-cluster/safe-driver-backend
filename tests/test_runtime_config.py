import config.config as config
import config.settings as settings
from database.firestore_helper import FirestoreHelper
import utils.utils as utils


def _firestore_config(**values):
    return {"raw_configurations": values}


def test_active_gps_simulator_port_is_not_replaced_by_firestore(monkeypatch):
    monkeypatch.delenv("GPS_SERIAL_PORT", raising=False)
    monkeypatch.setattr(config, "GPS_SERIAL_PORT", config.SIMULATED_GPS_SERIAL_PORT)
    monkeypatch.setattr(
        utils.os.path,
        "exists",
        lambda path: path == config.SIMULATED_GPS_SERIAL_PORT,
    )

    result = utils.update_local_config_from_firestore(
        _firestore_config(GPS_SERIAL_PORT=config.REAL_GPS_SERIAL_PORT)
    )

    assert result["success"] is True
    assert result["updated_count"] == 0
    assert config.GPS_SERIAL_PORT == config.SIMULATED_GPS_SERIAL_PORT


def test_explicit_gps_environment_port_is_not_replaced_by_firestore(monkeypatch):
    explicit_port = "/tmp/custom_gps"
    monkeypatch.setenv("GPS_SERIAL_PORT", explicit_port)
    monkeypatch.setattr(config, "GPS_SERIAL_PORT", explicit_port)
    monkeypatch.setattr(utils.os.path, "exists", lambda path: False)

    utils.update_local_config_from_firestore(
        _firestore_config(GPS_SERIAL_PORT=config.REAL_GPS_SERIAL_PORT)
    )

    assert config.GPS_SERIAL_PORT == explicit_port


def test_device_settings_are_updated_from_firestore(monkeypatch):
    monkeypatch.setattr(settings, "CAMERA_ROTATION", 0)
    monkeypatch.setattr(settings, "CAMERA_ZOOM", 1)
    monkeypatch.setattr(settings, "SYSTEM", "windows")

    result = utils.update_local_settings_from_firestore(
        {
            "CAMERA_ROTATION": 180,
            "CAMERA_ZOOM": 1,
            "SYSTEM": "linux",
            "last_updated": "ignored",
        }
    )

    assert result["success"] is True
    assert result["updated_count"] == 1
    assert settings.CAMERA_ROTATION == 180
    assert settings.CAMERA_ZOOM == 1
    assert settings.SYSTEM == "windows"
    assert not hasattr(settings, "last_updated")


def test_nested_tuple_config_is_firestore_safe_and_decoded(monkeypatch):
    value = ((0.45, 0.10), (0.60, 0.00))

    safe_value = FirestoreHelper._firestore_safe_value(value)

    assert safe_value == [
        {"sequence_values": [0.45, 0.10]},
        {"sequence_values": [0.60, 0.00]},
    ]

    monkeypatch.setattr(config, "BUZZER_ALERT_PATTERN", [])
    result = utils.update_local_config_from_firestore(
        _firestore_config(BUZZER_ALERT_PATTERN=safe_value)
    )

    assert result["success"] is True
    assert config.BUZZER_ALERT_PATTERN == [[0.45, 0.10], [0.60, 0.00]]
