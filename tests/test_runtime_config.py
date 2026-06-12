import config.config as config
import config.settings as settings
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
