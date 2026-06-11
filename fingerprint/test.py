import logging

from fingerprint.sensor import create_sensor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    f = create_sensor()

    logger.info('Sensor connected successfully!')
except Exception as e:
    print('Error:', e)
