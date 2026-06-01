from pyfingerprint.pyfingerprint import PyFingerprint
from utils import (
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
        sensor = PyFingerprint('/dev/serial0', 57600, 0xFFFFFFFF, 0x00000000)

        # if not sensor.verifyPassword():
        #     logger.info('Wrong password.')
        #     return False

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