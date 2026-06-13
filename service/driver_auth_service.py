import logging
import threading
import time

import cv2

import config.config as config
import config.settings as settings
import model.utilmethods as model_utils
import utils.utils as util
from database.firestore_helper import firestore_helper
from shared import behavior_queue, get_latest_camera_frame, reset_behavior_state
from utils.frame_transform import (
    apply_camera_rotation,
    apply_camera_zoom,
    normalize_camera_rotation,
    normalize_camera_zoom,
)


logger = logging.getLogger(__name__)


def _open_evidence_camera(camera_id):
    """Open the configured camera using the platform's supported strategy."""
    if settings.SYSTEM == "linux":
        # Lazy import avoids a module cycle because detect.py also uses the
        # driver-auth service.
        from model.detect import create_camera_capture

        capture, selected_camera_id, backend_name = create_camera_capture(
            camera_id,
            config.ALERT_EVIDENCE_SNAPSHOT_WIDTH,
            config.ALERT_EVIDENCE_SNAPSHOT_HEIGHT,
        )
        return capture, selected_camera_id, backend_name

    capture = cv2.VideoCapture(camera_id)
    return capture, camera_id, f"opencv:{camera_id}"


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
            self.fingerprint_disabled_movement_logged = False
            self.attendance_date = None
            self.attendance_document_id = None

    def is_verified(self):
        if not util.is_fingerprint_enabled():
            return True
        with self._lock:
            return self.verified_driver_id is not None

    def verified_driver(self):
        with self._lock:
            return self.verified_driver_id

    @staticmethod
    def _localized_message(messages):
        if isinstance(messages, dict):
            return messages.get(config.LANGUAGE) or messages.get("ENGLISH") or next(iter(messages.values()))
        return messages

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
            language_dependent=False,
        )
        logger.info("Requested driver fingerprint verification")
        return True

    def mark_verified(self, driver_id, driver_language=None):
        with self._lock:
            previous_driver_id = self.verified_driver_id
            self.verified_driver_id = driver_id
            self.unauthorized_attempts = 0
            self.unauthorized_cloud_sent = False
            self.unverified_movement_active = False
        if previous_driver_id != driver_id:
            reset_behavior_state(f"driver_verified:{driver_id}")
            attendance_result = firestore_helper.create_driver_attendance_signin(driver_id)
            if attendance_result.get("success"):
                with self._lock:
                    self.attendance_date = attendance_result.get("date")
                    self.attendance_document_id = attendance_result.get("document_id")
        if driver_language in ("ENGLISH", "SINHALA", "TAMIL"):
            config.LANGUAGE = driver_language
        model_utils.perform_voice_alerts(
            config.VOICE_ALERT_FINGERPRINT_VERIFIED,
            config.VOICE_ALERT_FINGERPRINT_VERIFIED_LABEL,
            language_dependent=False,
        )
        model_utils.perform_voice_alerts(
            self._localized_message(config.VOICE_ALERT_FINGERPRINT_SIGNOFF_INSTRUCTION),
            config.VOICE_ALERT_FINGERPRINT_SIGNOFF_INSTRUCTION_LABEL,
            language_dependent=True,
        )
        logger.info("Driver fingerprint verified: driver_id=%s", driver_id)

    def mark_signed_off(self):
        with self._lock:
            signed_off_driver_id = self.verified_driver_id
            attendance_date = self.attendance_date
            attendance_document_id = self.attendance_document_id
            self.verified_driver_id = None
            self.unauthorized_attempts = 0
            self.unauthorized_cloud_sent = False
            self.last_prompt_time = 0.0
            self.unverified_movement_active = False
            self.attendance_date = None
            self.attendance_document_id = None
        reset_behavior_state(f"driver_signed_off:{signed_off_driver_id or 'unknown'}")
        if signed_off_driver_id:
            firestore_helper.update_driver_attendance_signoff(
                signed_off_driver_id,
                attendance_date=attendance_date,
                attendance_document_id=attendance_document_id,
            )
        model_utils.perform_voice_alerts(
            self._localized_message(config.VOICE_ALERT_FINGERPRINT_SIGNOFF_SUCCESS),
            config.VOICE_ALERT_FINGERPRINT_SIGNOFF_SUCCESS_LABEL,
            language_dependent=True,
        )
        logger.info("Driver fingerprint signed off: driver_id=%s", signed_off_driver_id)
        return signed_off_driver_id

    def warn_wrong_signoff_driver(self):
        model_utils.perform_voice_alerts(
            self._localized_message(config.VOICE_ALERT_FINGERPRINT_SIGNOFF_WRONG_DRIVER),
            config.VOICE_ALERT_FINGERPRINT_SIGNOFF_WRONG_DRIVER_LABEL,
            language_dependent=True,
        )

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
                language_dependent=False,
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
        if not util.is_fingerprint_enabled():
            with self._lock:
                should_log = not self.fingerprint_disabled_movement_logged
                self.fingerprint_disabled_movement_logged = True
            if should_log:
                logger.warning(
                    "Unverified-driver movement alerts are disabled because fingerprint auth is disabled"
                )
            return False

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
            logger.warning(
                "%s: speed=%.2f km/h threshold=%.2f km/h",
                config.CONSOLE_UNVERIFIED_DRIVER_MOVEMENT,
                speed_kmh,
                config.DETECTION_ENABLE_SPEED_KMPH,
            )
            emitted = self._emit_security_alert(
                config.BEHAVIOR_UNVERIFIED_DRIVER_MOVEMENT,
                config.CONSOLE_UNVERIFIED_DRIVER_MOVEMENT,
                {"speed": speed_kmh, "threshold": config.DETECTION_ENABLE_SPEED_KMPH},
            )
            if emitted:
                with self._lock:
                    self.last_unverified_movement_alert_time = now
                return True
            logger.warning(
                "Unverified-driver movement alert will retry because it was not emitted"
            )
        return False

    @staticmethod
    def _capture_evidence_snapshot():
        """Briefly open the camera and capture evidence when no cached frame exists."""
        camera_id = config.ALERT_EVIDENCE_SNAPSHOT_CAMERA_ID
        capture = None
        try:
            capture, selected_camera_id, backend_name = _open_evidence_camera(camera_id)
            if capture is None or not capture.isOpened():
                logger.warning(
                    "Could not open camera %s for security-alert evidence",
                    camera_id,
                )
                return None

            for _ in range(config.ALERT_EVIDENCE_SNAPSHOT_READ_ATTEMPTS):
                success, frame = capture.read()
                if success and frame is not None:
                    camera_rotation = normalize_camera_rotation(settings.CAMERA_ROTATION)
                    camera_zoom = normalize_camera_zoom(settings.CAMERA_ZOOM)
                    frame = apply_camera_rotation(frame, camera_rotation)
                    frame = apply_camera_zoom(frame, camera_zoom)
                    logger.info(
                        "Captured security-alert evidence from camera %s using %s",
                        selected_camera_id,
                        backend_name,
                    )
                    return frame

            logger.warning(
                "Camera %s opened but did not return a security-alert evidence frame",
                selected_camera_id,
            )
            return None
        except Exception:
            logger.exception(
                "Failed to capture security-alert evidence from camera %s",
                camera_id,
            )
            return None
        finally:
            if capture is not None:
                capture.release()

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
            logger.info(
                "No cached camera frame for security alert; capturing one now: type=%s",
                event_type,
            )
            frame = self._capture_evidence_snapshot()
            if frame is None:
                logger.warning(
                    "Security alert deferred because camera evidence is unavailable: type=%s",
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
