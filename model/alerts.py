import json
import logging
import sys
import time
from typing import Any, Callable, Dict, Optional, TextIO

import config.config as config
import model.utilmethods as utils


ALERT_PRIORITY_ORDER = {
    "face_missing": 0,
    "microsleep": 1,
    "frequent_closures": 2,
    "yawn": 3,
    "drowsy": 4,
    "head_turn": 5,
}


class AlertManager:
    """Centralized alert event emitter and threshold alert tracker."""

    def __init__(
        self,
        logger: logging.Logger,
        now_provider: Callable[[], str],
        output_stream: TextIO,
        threshold_defaults: Optional[Dict[str, bool]] = None,
        buzzer_callback: Optional[Callable[[], None]] = None,
    ):
        self.logger = logger
        self.now_provider = now_provider
        self.output_stream = output_stream
        self.threshold_alert_sent = threshold_defaults.copy() if threshold_defaults else {}
        self.buzzer_callback = buzzer_callback
        self.last_event_time_by_type: Dict[str, float] = {}
        self.consecutive_count_by_type: Dict[str, int] = {}
        self.last_voice_alert_time_by_type: Dict[str, float] = {}
        self.last_buzzer_alert_time_by_type: Dict[str, float] = {}
        self.voice_alert_count_by_type: Dict[str, int] = {}
        self.buzzer_alert_count_by_type: Dict[str, int] = {}
        self._voice_cycle_state: Dict[str, Any] = {"cycle_id": None, "priority": None, "emitted": False}
        self._buzzer_cycle_state: Dict[str, Any] = {"cycle_id": None, "priority": None, "emitted": False}

    def _play_buzzer_beep(self) -> None:
        """Play a strong buzzer alarm pattern without any spoken message."""
        try:
            import winsound

            # High-frequency repeated beeps to quickly get driver attention.
            pattern = [
                (2200, 450),
                (2200, 450),
                (2200, 600),
            ]
            for frequency, duration_ms in pattern:
                winsound.Beep(frequency, duration_ms)
                time.sleep(0.1)
        except Exception:
            # Fallback terminal bell if winsound is unavailable.
            try:
                sys.stderr.write("\a")
                sys.stderr.flush()
            except Exception as exc:
                self.logger.warning(f"Buzzer beep fallback failed: {exc}")

    def _validate_alert_request(
        self,
        tag: str,
        event_type: str,
        message: str,
        behavior_data: Optional[Dict[str, Any]],
        current_count: Optional[int],
        threshold: Optional[int],
    ) -> bool:
        if not isinstance(tag, str) or not tag.strip():
            self.logger.error("Invalid alert tag. Expected a non-empty string.")
            return False

        if not isinstance(event_type, str) or not event_type.strip():
            self.logger.error("Invalid alert event_type. Expected a non-empty string.")
            return False

        if not isinstance(message, str) or not message.strip():
            self.logger.error("Invalid alert message. Expected a non-empty string.")
            return False

        if behavior_data is not None and not isinstance(behavior_data, dict):
            self.logger.error("Invalid alert behavior_data. Expected a dict.")
            return False

        if threshold is not None:
            if current_count is None:
                self.logger.error("Threshold validation failed. current_count is required when threshold is set.")
                return False
            if not isinstance(current_count, int) or not isinstance(threshold, int):
                self.logger.error("Threshold validation failed. current_count and threshold must be integers.")
                return False
            if threshold <= 0:
                self.logger.error("Threshold validation failed. threshold must be greater than zero.")
                return False

        return True

    def reset_event_state(self, policy_key: str) -> None:
        """Reset per-event policy state so alerts can trigger fresh after inactivity."""
        if not policy_key:
            return

        threshold_key = f"{policy_key}:threshold"
        self.threshold_alert_sent.pop(threshold_key, None)
        self.last_event_time_by_type.pop(policy_key, None)
        self.consecutive_count_by_type.pop(policy_key, None)
        self.last_voice_alert_time_by_type.pop(policy_key, None)
        self.last_buzzer_alert_time_by_type.pop(policy_key, None)
        self.voice_alert_count_by_type.pop(policy_key, None)
        self.buzzer_alert_count_by_type.pop(policy_key, None)

    def _update_event_activity(self, policy_key: str, now_ts: float) -> int:
        """Track consecutive occurrences and reset state after inactivity window."""
        last_event_time = self.last_event_time_by_type.get(policy_key)
        if last_event_time is not None and (now_ts - last_event_time) >= config.EVENT_COUNT_RESET_SEC:
            self.reset_event_state(policy_key)

        consecutive = self.consecutive_count_by_type.get(policy_key, 0) + 1
        self.consecutive_count_by_type[policy_key] = consecutive
        self.last_event_time_by_type[policy_key] = now_ts
        return consecutive

    def _channel_permits_emit(self, channel: str, cycle_id: Optional[int], priority_key: str) -> bool:
        """Allow only one alert per channel per cycle, honoring configured priority order."""
        if cycle_id is None:
            return True

        priority = ALERT_PRIORITY_ORDER.get(priority_key, 10_000)
        state = self._voice_cycle_state if channel == "voice" else self._buzzer_cycle_state

        if state["cycle_id"] != cycle_id:
            state["cycle_id"] = cycle_id
            state["priority"] = priority
            state["emitted"] = False

        if state["emitted"]:
            return False

        if priority > state["priority"]:
            return False

        state["priority"] = priority
        return True

    def send_behavior_to_parent(
        self,
        tag: str = "BEHAVIOR_EVENT",
        event_type: str = "behavior",
        message: str = "",
        event_time: Optional[str] = None,
        behavior_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send behavior data to parent process via stdout as JSON."""
        payload = {
            "tag": tag,
            "type": event_type,
            "message": message,
            "time": event_time or self.now_provider(),
            "data": behavior_data or {},
        }

        try:
            self.output_stream.write(f"BEHAVIOR_DATA:{json.dumps(payload)}\n")
            self.output_stream.flush()
        except Exception as exc:
            self.logger.error(f"Failed to send behavior data to parent: {exc}")

    def check_and_send_threshold_alert(
        self,
        *,
        tag: str,
        event_type: str,
        message: str,
        behavior_data: Optional[Dict[str, Any]] = None,
        policy_key: Optional[str] = None,
        cycle_id: Optional[int] = None,
        current_count: Optional[int] = None,
        threshold: Optional[int] = None,
        send_cloud: bool = True,
        trigger_voice: bool = False,
        voice_message: Optional[str] = None,
        trigger_buzzer: bool = False,
        buzzer_message: Optional[str] = None,
        timeframe_count: Optional[int] = None,
    ) -> None:
        """Common alert method with payload validation and channel routing."""
        if not self._validate_alert_request(tag, event_type, message, behavior_data, current_count, threshold):
            return

        policy_key = policy_key or event_type
        payload_data = behavior_data or {}
        event_time = self.now_provider()
        now_ts = time.time()
        consecutive_count = self._update_event_activity(policy_key, now_ts)

        threshold_reached = True
        threshold_key = None

        if threshold is not None:
            threshold_reached = current_count >= threshold
            threshold_key = f"{policy_key}:threshold"
            payload_data = {
                **payload_data,
                "event_type": event_type,
                "count": current_count,
                "threshold": threshold,
            }

        if send_cloud and threshold is not None:
            if not threshold_reached:
                self.logger.debug(
                    "Cloud alert suppressed for %s: count %s is below threshold %s",
                    policy_key,
                    current_count,
                    threshold,
                )
                send_cloud = False
            elif self.threshold_alert_sent.get(threshold_key):
                self.logger.debug(
                    "Cloud alert suppressed for %s: threshold alert already sent in current cycle",
                    policy_key,
                )
                send_cloud = False
            else:
                self.threshold_alert_sent[threshold_key] = True

        if send_cloud:
            self.send_behavior_to_parent(
                tag=tag,
                event_type=event_type,
                message=message,
                event_time=event_time,
                behavior_data=payload_data,
            )

        if trigger_voice:
            if not self._channel_permits_emit("voice", cycle_id, policy_key):
                trigger_voice = False

        if trigger_voice:
            allow_voice = consecutive_count >= config.VOICE_ALERT_CONSECUTIVE_EVENT_THRESH
            buzzer_used = self.buzzer_alert_count_by_type.get(policy_key, 0)
            voice_used = self.voice_alert_count_by_type.get(policy_key, 0)
            last_voice = self.last_voice_alert_time_by_type.get(policy_key)
            voice_cooldown_ok = last_voice is None or (now_ts - last_voice) >= config.VOICE_ALERT_COOLDOWN_SEC

            if buzzer_used >= config.MAXIMUM_BUZZER_ALERTS_PER_TYPE and allow_voice and voice_cooldown_ok and voice_used < config.MAXIMUM_VOICE_ALERTS_PER_TYPE:
                # voice_text = voice_message or message
                voice_text = self.get_voice_msg_by_level(event_type, level=voice_used + 1)
                if isinstance(voice_text, str) and voice_text.strip():
                    utils.perform_voice_alerts(voice_text)
                    self.voice_alert_count_by_type[policy_key] = voice_used + 1
                    self.last_voice_alert_time_by_type[policy_key] = now_ts
                    self._voice_cycle_state["emitted"] = True
                else:
                    self.logger.warning(f"Voice alert skipped due to empty message for event type: {event_type}")

        if trigger_buzzer:
            if not self._channel_permits_emit("buzzer", cycle_id, policy_key):
                trigger_buzzer = False

        if trigger_buzzer:
            allow_buzzer = consecutive_count >= config.BUZZER_ALERT_CONSECUTIVE_EVENT_THRESH
            buzzer_used = self.buzzer_alert_count_by_type.get(policy_key, 0)
            last_buzzer = self.last_buzzer_alert_time_by_type.get(policy_key)
            buzzer_cooldown_ok = last_buzzer is None or (now_ts - last_buzzer) >= config.BUZZER_ALERT_COOLDOWN_SEC

            if allow_buzzer and buzzer_cooldown_ok and buzzer_used < config.MAXIMUM_BUZZER_ALERTS_PER_TYPE:
                if self.buzzer_callback:
                    self.buzzer_callback()
                else:
                    self._play_buzzer_beep()
                self.buzzer_alert_count_by_type[policy_key] = buzzer_used + 1
                self.last_buzzer_alert_time_by_type[policy_key] = now_ts
                self._buzzer_cycle_state["emitted"] = True
    
    def get_voice_msg_by_level(self, event_type: str, level: int) -> str:
        """Return appropriate voice message based on event type and severity level."""

        # Behavior Data Message Types
        # BEHAVIOR_FREQUENT_CLOSURES = 'frequent_closures'
        # BEHAVIOR_MICROSLEEP = 'microsleep'
        # BEHAVIOR_YAWN = 'yawn'
        # BEHAVIOR_DROWSY = 'drowsy'
        # BEHAVIOR_PERCLOS_REACHED = 'perclos_threshold_reached'
        # BEHAVIOR_DISTRACTION = 'distraction'
        # BEHAVIOR_HEAD_TURN = "head_turn"
        # BEHAVIOR_MOBILE_USE = 'mobile_use'
        # BEHAVIOR_SMOKING = 'smoking'
        # BEHAVIOR_DRINKING = 'drinking'

        if event_type == config.BEHAVIOR_DROWSY:
            if level == 1:
                return config.VOICE_ALERT_DROWSY
            if level == 2:
                return config.VOICE_ALERT_DROWSY_L2
            if level == 3:
                return config.VOICE_ALERT_DROWSY_L3
            
        if event_type == config.BEHAVIOR_DISTRACTION:
            if level == 1:
                return config.VOICE_ALERT_DISTRACTION
            if level == 2:
                return config.VOICE_ALERT_DISTRACTION_L2
            if level == 3:
                return config.VOICE_ALERT_DISTRACTION_L3
            
        if event_type == config.BEHAVIOR_HEAD_TURN:
            if level == 1:
                return config.VOICE_ALERT_HEAD_TURN
            if level == 2:
                return config.VOICE_ALERT_HEAD_TURN_L2
            if level == 3:
                return config.VOICE_ALERT_HEAD_TURN_L3
            
        if event_type == config.BEHAVIOR_PERCLOS_REACHED:
            if level == 1:
                return config.VOICE_ALERT_PERCLOS
            if level == 2:
                return config.VOICE_ALERT_PERCLOS_L2
            if level == 3:
                return config.VOICE_ALERT_PERCLOS_L3
            
        if event_type == config.BEHAVIOR_YAWN:
            if level == 1:
                return config.VOICE_ALERT_YAWNING
            if level == 2:
                return config.VOICE_ALERT_YAWNING_L2
            if level == 3:
                return config.VOICE_ALERT_YAWNING_L3
            
        if event_type == config.BEHAVIOR_MICROSLEEP:
            if level == 1:
                return config.VOICE_ALERT_MICROSLEEP
            if level == 2:
                return config.VOICE_ALERT_MICROSLEEP_L2
            if level == 3:
                return config.VOICE_ALERT_MICROSLEEP_L3
            
        if event_type == config.BEHAVIOR_FREQUENT_CLOSURES:
            if level == 1:
                return config.VOICE_ALERT_FREQUENT_CLOSURES
            if level == 2:
                return config.VOICE_ALERT_FREQUENT_CLOSURES_L2
            if level == 3:
                return config.VOICE_ALERT_FREQUENT_CLOSURES_L3
            
        if event_type == config.BEHAVIOR_MOBILE_USE:
            if level == 1:
                return config.VOICE_ALERT_PHONE
            if level == 2:
                return config.VOICE_ALERT_PHONE_L2
            if level == 3:
                return config.VOICE_ALERT_PHONE_L3
            
        if event_type == config.BEHAVIOR_SMOKING:
            if level == 1:
                return config.VOICE_ALERT_SMOKING
            if level == 2:
                return config.VOICE_ALERT_SMOKING_L2
            if level == 3:
                return config.VOICE_ALERT_SMOKING_L3
            
        if event_type == config.BEHAVIOR_DRINKING:
            if level == 1:
                return config.VOICE_ALERT_DRINKING
            if level == 2:
                return config.VOICE_ALERT_DRINKING_L2
            if level == 3:
                return config.VOICE_ALERT_DRINKING_L3
            
        return config.VOICE_ALERT_DEFAULT