import logging
import time

import config.config as config


logger = logging.getLogger(__name__)

_motor = None
_checked = False


def _get_motor():
    """Initialize GPIO vibration motor only when an alert needs it."""
    global _motor, _checked

    if not config.ENABLE_VIBRATION_ALERTS:
        return None

    if _checked:
        return _motor

    _checked = True

    try:
        from gpiozero import OutputDevice

        _motor = OutputDevice(
            config.VIBRATOR_GPIO_PIN,
            active_high=config.VIBRATOR_ACTIVE_HIGH,
            initial_value=False,
        )
        logger.info(
            "Vibration motor initialized on GPIO %s, active_high=%s",
            config.VIBRATOR_GPIO_PIN,
            config.VIBRATOR_ACTIVE_HIGH,
        )
    except Exception as exc:
        _motor = None
        logger.warning("Vibration motor unavailable: %s", exc)

    return _motor


def vibrate_alert(duration_sec=None):
    """Run the vibration motor briefly for a behavior alert."""
    motor = _get_motor()
    if motor is None:
        return

    duration = config.VIBRATOR_DURATION_SEC if duration_sec is None else float(duration_sec)

    try:
        motor.on()
        time.sleep(max(0.0, duration))
    except Exception as exc:
        logger.warning("Vibration alert failed: %s", exc)
    finally:
        try:
            motor.off()
        except Exception as exc:
            logger.warning("Failed to stop vibration motor: %s", exc)


def shutdown_vibrator():
    """Turn off and release the vibration motor GPIO device."""
    global _motor, _checked

    if _motor is None:
        return

    try:
        _motor.off()
        _motor.close()
    except Exception as exc:
        logger.warning("Failed to close vibration motor: %s", exc)
    finally:
        _motor = None
        _checked = False
