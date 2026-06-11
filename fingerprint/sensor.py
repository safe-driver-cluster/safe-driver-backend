import logging
import os

import config.config as config
from pyfingerprint.pyfingerprint import PyFingerprint


logger = logging.getLogger(__name__)


def _candidate_ports():
    ports = []
    for port in (config.FINGERPRINT_SERIAL_PORT, *config.FINGERPRINT_FALLBACK_SERIAL_PORTS):
        if port and port not in ports:
            ports.append(port)
    return ports


def _flush_sensor_input(sensor):
    serial_connection = getattr(sensor, "_PyFingerprint__serial", None)
    if serial_connection is None:
        return

    try:
        serial_connection.reset_input_buffer()
    except AttributeError:
        try:
            serial_connection.flushInput()
        except Exception:
            pass
    except Exception:
        pass


def close_sensor(sensor):
    """Close the pyfingerprint serial connection if it is open."""
    serial_connection = getattr(sensor, "_PyFingerprint__serial", None)
    if serial_connection is None:
        return

    try:
        if serial_connection.isOpen():
            serial_connection.close()
    except Exception:
        pass


def is_packet_header_error(exc):
    message = str(exc).lower()
    return (
        "valid header" in message
        or ("unsupported operand type" in message and "bytes" in message)
    )


def create_sensor(verify_password=True, exclude_ports=None):
    """Create a fingerprint sensor, trying configured Raspberry Pi serial ports."""
    failures = []
    excluded = set(exclude_ports or ())

    for port in _candidate_ports():
        if port in excluded:
            failures.append(f"{port}: skipped after packet errors")
            continue

        if port.startswith("/dev/") and not os.path.exists(port):
            failures.append(f"{port}: not found")
            continue

        try:
            sensor = PyFingerprint(
                port,
                config.FINGERPRINT_BAUDRATE,
                config.FINGERPRINT_ADDRESS,
                config.FINGERPRINT_PASSWORD,
            )
            _flush_sensor_input(sensor)

            if verify_password and not sensor.verifyPassword():
                failures.append(f"{port}: password verification failed")
                close_sensor(sensor)
                continue

            _flush_sensor_input(sensor)
            sensor.safe_driver_port = port
            logger.info("Fingerprint sensor connected on %s", port)
            return sensor
        except Exception as exc:
            failures.append(f"{port}: {exc}")

    raise RuntimeError("Fingerprint sensor unavailable. Tried " + "; ".join(failures))
