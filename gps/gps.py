import logging
from math import atan2, cos, radians, sin, sqrt
import os
import time

import pynmea2
import serial

import config.config as config
from database import db_helper
from database.firestore_helper import firestore_helper
import model.utilmethods as model_utils
from service.model_service import get_mac_address_alternative
from service.driver_auth_service import driver_auth_service


logger = logging.getLogger(__name__)

DEFAULT_GPS_PORT = "/dev/ttyAMA5"
DEFAULT_GPS_BAUDRATE = 9600


def resolve_gps_port(configured_port=None):
    """Select the currently available GPS port without requiring a restart."""
    environment_port = os.getenv("GPS_SERIAL_PORT")
    if environment_port:
        return environment_port

    simulator_port = config.SIMULATED_GPS_SERIAL_PORT
    if simulator_port and os.path.exists(simulator_port):
        return simulator_port

    if configured_port and configured_port != simulator_port:
        return configured_port

    return config.REAL_GPS_SERIAL_PORT


def distance_meters(latitude_a, longitude_a, latitude_b, longitude_b):
    """Calculate great-circle distance between two coordinates in meters."""
    earth_radius_meters = 6_371_000
    lat_a = radians(latitude_a)
    lat_b = radians(latitude_b)
    delta_lat = radians(latitude_b - latitude_a)
    delta_lon = radians(longitude_b - longitude_a)
    value = (
        sin(delta_lat / 2) ** 2
        + cos(lat_a) * cos(lat_b) * sin(delta_lon / 2) ** 2
    )
    return earth_radius_meters * 2 * atan2(sqrt(value), sqrt(1 - value))


class HazardZoneMonitor:
    """Cache Firestore hazards and warn once whenever the bus enters a zone."""

    def __init__(self, hazard_provider=None, voice_callback=None, time_provider=None):
        self.hazard_provider = hazard_provider or firestore_helper.get_hazard_zones
        self.voice_callback = voice_callback or model_utils.perform_voice_alerts
        self.time_provider = time_provider or time.time
        self.hazards = []
        self.active_hazard_ids = set()
        self.last_refresh_time = 0.0
        self.loaded_once = False

    @staticmethod
    def _voice_for_hazard(hazard):
        hazard_type = str(
            hazard.get("type") or config.DEFAULT_HAZARD_ALERT_TYPE
        ).strip().lower()
        hazard_type = config.HAZARD_TYPE_ALIASES.get(hazard_type, hazard_type)
        alert_config = config.VOICE_ALERT_HAZARD_TYPES.get(
            hazard_type,
            config.VOICE_ALERT_HAZARD_TYPES[config.DEFAULT_HAZARD_ALERT_TYPE],
        )
        language = (
            config.LANGUAGE
            if config.LANGUAGE in alert_config["messages"]
            else "ENGLISH"
        )
        return alert_config["messages"][language], alert_config["label"], hazard_type

    def _refresh_if_due(self):
        if config.HAZARD_FETCH_ONCE and self.loaded_once:
            return

        now = self.time_provider()
        if (
            not config.HAZARD_FETCH_ONCE
            and
            self.last_refresh_time
            and now - self.last_refresh_time < config.HAZARD_REFRESH_INTERVAL_SEC
        ):
            return

        hazards = self.hazard_provider()
        self.last_refresh_time = now
        if hazards is not None:
            self.hazards = hazards
            self.loaded_once = True
            valid_ids = {hazard["id"] for hazard in hazards}
            self.active_hazard_ids.intersection_update(valid_ids)
            logger.info("Hazard monitor loaded %s zones", len(hazards))

    def check_location(self, latitude, longitude):
        if not config.ENABLE_HAZARD_WARNINGS:
            return []

        self._refresh_if_due()
        entered = []

        for hazard in self.hazards:
            distance = distance_meters(
                latitude,
                longitude,
                hazard["latitude"],
                hazard["longitude"],
            )
            hazard_id = hazard["id"]

            if distance <= hazard["radius"]:
                if hazard_id not in self.active_hazard_ids:
                    self.active_hazard_ids.add(hazard_id)
                    entered.append(hazard_id)
                    message, label, hazard_type = self._voice_for_hazard(hazard)
                    self.voice_callback(message, label)
                    logger.warning(
                        "Entered hazard zone: id=%s type=%s distance=%.1fm radius=%.1fm",
                        hazard_id,
                        hazard_type,
                        distance,
                        hazard["radius"],
                    )
            elif distance > hazard["radius"] + config.HAZARD_EXIT_BUFFER_METERS:
                if hazard_id in self.active_hazard_ids:
                    self.active_hazard_ids.remove(hazard_id)
                    logger.info("Exited hazard zone: id=%s distance=%.1fm", hazard_id, distance)

        return entered


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
    """Read GPS NMEA data and switch between simulator and real GPS as needed."""
    last_push_time = 0.0
    device_mac = device_mac or get_mac_address_alternative()
    hazard_monitor = HazardZoneMonitor()

    while not stop_event.is_set():
        gps_serial = None
        active_port = resolve_gps_port(port)
        try:
            gps_serial = serial.Serial(active_port, baudrate=baudrate, timeout=1)
            logger.info("GPS worker connected on %s at %s baud", active_port, baudrate)

            while not stop_event.is_set():
                raw = gps_serial.readline()
                preferred_port = resolve_gps_port(port)
                if preferred_port != active_port:
                    logger.info(
                        "GPS source changed: %s -> %s; reconnecting",
                        active_port,
                        preferred_port,
                    )
                    break

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
                driver_auth_service.report_speed(data["speed"])
                hazard_monitor.check_location(data["latitude"], data["longitude"])

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
            logger.error("GPS serial error on %s: %s", active_port, e)
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
