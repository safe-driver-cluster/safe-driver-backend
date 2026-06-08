import logging
import time

import pynmea2
import serial

import config.config as config
from database import db_helper
from service.model_service import get_mac_address_alternative


logger = logging.getLogger(__name__)

DEFAULT_GPS_PORT = "/dev/ttyAMA5"
DEFAULT_GPS_BAUDRATE = 9600


def _parse_rmc_line(line: str):
    if not line.startswith(("$GPRMC", "$GNRMC")):
        return None

    try:
        msg = pynmea2.parse(line)
    except pynmea2.ParseError:
        return None

    if msg.status != "A":
        return None

    speed_kmh = float(msg.spd_over_grnd) * 1.852 if msg.spd_over_grnd else 0.0
    if speed_kmh < config.SPEED_THRESHOLD:
        speed_kmh = 0.0

    return {
        "latitude": round(msg.latitude, 6),
        "longitude": round(msg.longitude, 6),
        "speed": round(speed_kmh, 2),
        "active_speed": speed_kmh > config.DETECTION_ENABLE_SPEED_KMPH,
        "timestamp": int(time.time()),
    }


def run_gps_loop(stop_event, device_mac=None, port=DEFAULT_GPS_PORT, baudrate=DEFAULT_GPS_BAUDRATE):
    """Read GPS NMEA data and update Firebase at the configured interval."""
    last_push_time = 0.0
    device_mac = device_mac or get_mac_address_alternative()

    while not stop_event.is_set():
        gps_serial = None
        try:
            gps_serial = serial.Serial(port, baudrate=baudrate, timeout=1)
            logger.info("GPS worker connected on %s at %s baud", port, baudrate)

            while not stop_event.is_set():
                raw = gps_serial.readline()
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                data = _parse_rmc_line(line)
                if data is None:
                    continue

                # Detection uses the latest valid GPS speed immediately. Firebase
                # updates remain limited to PUSH_INTERVAL.
                config.CURRENT_SPEED = data["speed"]
                config.CURRENT_SPEED_UPDATED_AT = time.time()

                now = time.time()
                time_elapsed = (now - last_push_time) >= config.PUSH_INTERVAL

                if time_elapsed:
                    last_push_time = now
                    result = db_helper.update_device_gps(device_mac, data)

                    if result.get("success"):
                        if data["active_speed"]:
                            logger.info(
                                "GPS pushed: %.2f km/h, detection active",
                                data["speed"],
                            )
                        else:
                            logger.info(
                                "GPS pushed: %.2f km/h, detection paused",
                                data["speed"],
                            )
                    else:
                        logger.warning("GPS Firebase update failed: %s", result.get("message"))

        except serial.SerialException as e:
            logger.error("GPS serial error on %s: %s", port, e)
        except Exception:
            logger.exception("GPS worker failed")
        finally:
            if gps_serial is not None:
                try:
                    gps_serial.close()
                except Exception:
                    pass

        if not stop_event.is_set():
            logger.info("GPS reconnecting in %s seconds", config.GPS_RECONNECT_INTERVAL)
            stop_event.wait(config.GPS_RECONNECT_INTERVAL)

    logger.info("GPS worker stopped")


if __name__ == "__main__":
    import threading

    stop = threading.Event()
    try:
        run_gps_loop(stop)
    except KeyboardInterrupt:
        stop.set()
