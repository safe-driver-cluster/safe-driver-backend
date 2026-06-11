import logging
import threading
from contextlib import contextmanager


logger = logging.getLogger(__name__)

fingerprint_pause_event = threading.Event()
fingerprint_sensor_lock = threading.Lock()


@contextmanager
def exclusive_fingerprint_operation(name):
    """Pause live verification and reserve the fingerprint sensor serial port."""
    logger.info("Requesting exclusive fingerprint operation: %s", name)
    fingerprint_pause_event.set()
    fingerprint_sensor_lock.acquire()
    try:
        logger.info("Exclusive fingerprint operation started: %s", name)
        yield
    finally:
        fingerprint_sensor_lock.release()
        fingerprint_pause_event.clear()
        logger.info("Exclusive fingerprint operation finished: %s", name)
