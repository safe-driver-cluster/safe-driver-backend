from pyfingerprint.pyfingerprint import PyFingerprint

try:
    f = PyFingerprint('/dev/serial0', 57600, 0xFFFFFFFF, 0x00000000)

    if f.verifyPassword():
        logger.info('Sensor connected successfully!')
    else:
        logger.info('Wrong password!')
except Exception as e:
    print('Error:', e)