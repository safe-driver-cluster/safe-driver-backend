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


def create_sensor(verify_password=True):
    """Create a fingerprint sensor, trying configured Raspberry Pi serial ports."""
    failures = []

    for port in _candidate_ports():
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

            if verify_password and not sensor.verifyPassword():
                failures.append(f"{port}: password verification failed")
                continue

            logger.info("Fingerprint sensor connected on %s", port)
            return sensor
        except Exception as exc:
            failures.append(f"{port}: {exc}")

    raise RuntimeError("Fingerprint sensor unavailable. Tried " + "; ".join(failures))
