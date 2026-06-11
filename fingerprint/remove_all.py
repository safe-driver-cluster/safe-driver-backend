from fingerprint.sensor import create_sensor
from fingerprint.utils import (
    announce,
    get_scanner_id,
    build_fingerprint_template_id,
    wait_for_finger,
    init_audio,
)
import logging

try:
    from gtts import gTTS
except Exception:
    gTTS = None

logger = logging.getLogger(__name__)


def delete_all_fingerprints():
    try:
        sensor = create_sensor()

        sensor.clearDatabase()

        logger.info('All fingerprints deleted successfully.')
        return True

    except Exception as e:
        print('Exception:', e)
        return False

# ----------------------------------
# Execute deletion all fingerprints
# ----------------------------------
if __name__ == '__main__':
    success = delete_all_fingerprints()
    if success:
        announce('Delete all fingerprints successfully.')
    else:
        announce('Operation failed!')
