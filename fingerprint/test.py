if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

from fingerprint.sensor import create_sensor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    f = create_sensor()

    logger.info('Sensor connected successfully!')
except Exception as e:
    print('Error:', e)
