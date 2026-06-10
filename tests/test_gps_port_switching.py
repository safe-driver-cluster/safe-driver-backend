import config.config as config
import gps.gps as gps


def test_simulator_port_is_preferred_while_active(monkeypatch):
    monkeypatch.delenv("GPS_SERIAL_PORT", raising=False)
    monkeypatch.setattr(
        gps.os.path,
        "exists",
        lambda path: path == config.SIMULATED_GPS_SERIAL_PORT,
    )

    assert gps.resolve_gps_port(config.SIMULATED_GPS_SERIAL_PORT) == config.SIMULATED_GPS_SERIAL_PORT


def test_real_port_is_selected_after_simulator_stops(monkeypatch):
    monkeypatch.delenv("GPS_SERIAL_PORT", raising=False)
    monkeypatch.setattr(gps.os.path, "exists", lambda path: False)

    assert gps.resolve_gps_port(config.SIMULATED_GPS_SERIAL_PORT) == config.REAL_GPS_SERIAL_PORT


def test_explicit_environment_port_disables_automatic_switching(monkeypatch):
    monkeypatch.setenv("GPS_SERIAL_PORT", "/tmp/custom_gps")
    monkeypatch.setattr(
        gps.os.path,
        "exists",
        lambda path: path == config.SIMULATED_GPS_SERIAL_PORT,
    )

    assert gps.resolve_gps_port(config.REAL_GPS_SERIAL_PORT) == "/tmp/custom_gps"
