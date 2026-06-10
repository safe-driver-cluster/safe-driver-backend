import logging
import threading
import time

import cv2

import config.config as config
import model.utilmethods as model_utils
from shared import behavior_queue, get_latest_camera_frame


logger = logging.getLogger(__name__)


class DriverAuthService:
    """Coordinate fingerprint verification and security-alert escalation."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.verified_driver_id = None
            self.unauthorized_attempts = 0
            self.unauthorized_cloud_sent = False
            self.last_prompt_time = 0.0
            self.last_unverified_movement_alert_time = 0.0
            self.unverified_movement_active = False

    def is_verified(self):
        if not config.ENABLE_FINGERPRINT:
            return True
        with self._lock:
            return self.verified_driver_id is not None

    def verified_driver(self):
        with self._lock:
            return self.verified_driver_id

    def request_verification_if_due(self):
        if self.is_verified():
            return False
        now = time.time()
        with self._lock:
            if now - self.last_prompt_time < config.FINGERPRINT_VERIFICATION_PROMPT_INTERVAL_SEC:
                return False
            self.last_prompt_time = now
        model_utils.perform_voice_alerts(
            config.VOICE_ALERT_FINGERPRINT_REQUIRED,
            config.VOICE_ALERT_FINGERPRINT_REQUIRED_LABEL,
        )
        logger.info("Requested driver fingerprint verification")
        return True

    def mark_verified(self, driver_id, driver_language=None):
        with self._lock:
            self.verified_driver_id = driver_id
            self.unauthorized_attempts = 0
            self.unauthorized_cloud_sent = False
            self.unverified_movement_active = False
        if driver_language in ("ENGLISH", "SINHALA", "TAMIL"):
            config.LANGUAGE = driver_language
        model_utils.perform_voice_alerts(
            config.VOICE_ALERT_FINGERPRINT_VERIFIED,
            config.VOICE_ALERT_FINGERPRINT_VERIFIED_LABEL,
        )
        logger.info("Driver fingerprint verified: driver_id=%s", driver_id)

    def record_unauthorized_attempt(self):
        with self._lock:
            self.unauthorized_attempts += 1
            attempt = self.unauthorized_attempts
            should_warn = attempt <= config.UNAUTHORIZED_FINGERPRINT_WARNING_LIMIT
            should_cloud = attempt > config.UNAUTHORIZED_FINGERPRINT_WARNING_LIMIT and not self.unauthorized_cloud_sent
        if should_warn:
            model_utils.perform_voice_alerts(
                config.VOICE_ALERT_UNAUTHORIZED_DRIVER,
                config.VOICE_ALERT_UNAUTHORIZED_DRIVER_LABEL,
            )
            logger.warning("Unauthorized fingerprint attempt: %s/%s", attempt, config.UNAUTHORIZED_FINGERPRINT_WARNING_LIMIT)
        elif should_cloud:
            emitted = self._emit_security_alert(
                config.BEHAVIOR_UNAUTHORIZED_DRIVER,
                config.CONSOLE_UNAUTHORIZED_DRIVER,
                {"unauthorized_attempts": attempt},
            )
            if emitted:
                with self._lock:
                    self.unauthorized_cloud_sent = True
        return attempt

    def report_speed(self, speed_kmh):
        if self.is_verified():
            return False
        if speed_kmh <= config.DETECTION_ENABLE_SPEED_KMPH:
            with self._lock:
                self.unverified_movement_active = False
            return False
        now = time.time()
        with self._lock:
            cooldown_ok = now - self.last_unverified_movement_alert_time >= config.UNVERIFIED_MOVEMENT_ALERT_COOLDOWN_SEC
            should_alert = not self.unverified_movement_active or cooldown_ok
            self.unverified_movement_active = True
        if should_alert:
            emitted = self._emit_security_alert(
                config.BEHAVIOR_UNVERIFIED_DRIVER_MOVEMENT,
                config.CONSOLE_UNVERIFIED_DRIVER_MOVEMENT,
                {"speed": speed_kmh, "threshold": config.DETECTION_ENABLE_SPEED_KMPH},
            )
            if emitted:
                with self._lock:
                    self.last_unverified_movement_alert_time = now
                return True
        return False

    def _emit_security_alert(self, event_type, message, data):
        payload = {
            "tag": "DRIVER_SECURITY_EVENT",
            "type": event_type,
            "message": message,
            "time": model_utils.now(),
            "data": data,
            "driver": self.verified_driver() or "",
        }
        frame = get_latest_camera_frame()
        if config.ENABLE_ALERT_EVIDENCE and frame is None:
            logger.warning(
                "Security alert deferred until camera evidence is available: type=%s",
                event_type,
            )
            return False
        if config.ENABLE_ALERT_EVIDENCE:
            encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.ALERT_EVIDENCE_JPEG_QUALITY])
            if not encoded:
                logger.warning("Security alert deferred because evidence encoding failed: type=%s", event_type)
                return False
            payload["_evidence_jpeg"] = buffer.tobytes()
        behavior_queue.put(payload)
        logger.warning("Driver security cloud alert emitted: type=%s", event_type)
        return True


driver_auth_service = DriverAuthService()
