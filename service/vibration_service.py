import atexit
import logging
import threading
import time
from typing import Optional

import config.config as config


logger = logging.getLogger(__name__)


class VibrationService:
    """Non-blocking GPIO18 controller for the 5V vibration motor module."""

    def __init__(
        self,
        gpio_pin: int = config.VIBRATION_GPIO_PIN,
        enabled: bool = config.ENABLE_VIBRATION_ALERTS,
    ):
        self.gpio_pin = gpio_pin
        self.enabled = enabled
        self._gpio = None
        self._initialized = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._current_level: Optional[int] = None
        atexit.register(self.cleanup)

    def _initialize_gpio(self) -> bool:
        if not self.enabled:
            return False

        if self._initialized:
            return True

        try:
            import RPi.GPIO as GPIO

            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            self._initialized = True
            logger.info("Vibration motor initialized on GPIO%s", self.gpio_pin)
            return True
        except Exception as exc:
            self.enabled = False
            logger.warning("Vibration motor disabled: GPIO initialization failed: %s", exc)
            return False

    def _write(self, is_on: bool) -> None:
        if not self._initialize_gpio():
            return

        self._gpio.output(self.gpio_pin, self._gpio.HIGH if is_on else self._gpio.LOW)

    def start(self) -> None:
        """Turn vibration on continuously."""
        with self._lock:
            self.stop(join=False)
            self._current_level = config.VIBRATION_LEVEL_4
            self._write(True)

    def stop(self, join: bool = True) -> None:
        """Stop any active vibration pattern and drive GPIO18 LOW."""
        self._stop_event.set()
        worker = self._worker
        if join and worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=config.VIBRATION_WORKER_JOIN_TIMEOUT_SEC)

        self._worker = None
        self._current_level = None
        if self._initialized:
            self._write(False)

    def run_level_1(self) -> None:
        self.run_pattern(config.VIBRATION_LEVEL_1)

    def run_level_2(self) -> None:
        self.run_pattern(config.VIBRATION_LEVEL_2)

    def run_level_3(self) -> None:
        self.run_pattern(config.VIBRATION_LEVEL_3)

    def run_level_4(self) -> None:
        self.run_pattern(config.VIBRATION_LEVEL_4)

    def run_pattern(self, level: int) -> None:
        """Start the configured vibration pattern for an alert level."""
        if not self.enabled:
            return

        with self._lock:
            if self._current_level == level and self._worker and self._worker.is_alive():
                return

            self.stop(join=True)
            self._stop_event.clear()
            self._current_level = level

            pattern = config.VIBRATION_PATTERNS.get(level)
            if pattern is None:
                logger.warning("Unknown vibration alert level: %s", level)
                return

            on_sec, off_sec = pattern
            if off_sec is None:
                self._write(True)
                return

            self._worker = threading.Thread(
                target=self._pattern_loop,
                args=(level, on_sec, off_sec),
                name=f"VibrationLevel{level}",
                daemon=True,
            )
            self._worker.start()

    def _pattern_loop(self, level: int, on_sec: float, off_sec: float) -> None:
        while not self._stop_event.is_set():
            self._write(True)
            if self._stop_event.wait(on_sec):
                break
            self._write(False)
            self._stop_event.wait(off_sec)

        self._write(False)
        logger.debug("Vibration level %s worker stopped", level)

    def cleanup(self) -> None:
        """Release GPIO resources during application shutdown."""
        try:
            self.stop(join=True)
            if self._gpio and self._initialized:
                self._gpio.cleanup(self.gpio_pin)
        except Exception as exc:
            logger.warning("Vibration cleanup failed: %s", exc)
        finally:
            self._initialized = False
            self._gpio = None


vibration_service = VibrationService()
