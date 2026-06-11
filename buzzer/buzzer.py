import logging
import time

import config.config as config


logger = logging.getLogger(__name__)

_buzzer = None
_checked = False


def _get_buzzer():
    """Initialize the GPIO buzzer only when an alert needs it."""
    global _buzzer, _checked

    if not config.ENABLE_BUZZER_ALERTS:
        return None

    if _checked:
        return _buzzer

    _checked = True

    try:
        from gpiozero import OutputDevice

        _buzzer = OutputDevice(
            config.BUZZER_GPIO_PIN,
            active_high=config.BUZZER_ACTIVE_HIGH,
            initial_value=False,
        )
        logger.info(
            "Buzzer initialized on GPIO %s, active_high=%s",
            config.BUZZER_GPIO_PIN,
            config.BUZZER_ACTIVE_HIGH,
        )
    except Exception as exc:
        _buzzer = None
        logger.warning("GPIO buzzer unavailable: %s", exc)

    return _buzzer


def buzzer_alert():
    """Play the configured alert pattern on an active GPIO buzzer."""
    buzzer = _get_buzzer()
    if buzzer is None:
        return False

    try:
        for on_duration, off_duration in config.BUZZER_ALERT_PATTERN:
            buzzer.on()
            time.sleep(max(0.0, float(on_duration)))
            buzzer.off()
            time.sleep(max(0.0, float(off_duration)))
        return True
    except Exception as exc:
        logger.warning("GPIO buzzer alert failed: %s", exc)
        return False
    finally:
        try:
            buzzer.off()
        except Exception as exc:
            logger.warning("Failed to stop GPIO buzzer: %s", exc)


def shutdown_buzzer():
    """Turn off and release the GPIO buzzer."""
    global _buzzer, _checked

    if _buzzer is None:
        return

    try:
        _buzzer.off()
        _buzzer.close()
    except Exception as exc:
        logger.warning("Failed to close GPIO buzzer: %s", exc)
    finally:
        _buzzer = None
        _checked = False
