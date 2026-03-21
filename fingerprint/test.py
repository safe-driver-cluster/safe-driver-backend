from pyfingerprint.pyfingerprint import PyFingerprint

try:
    f = PyFingerprint('/dev/serial0', 57600, 0xFFFFFFFF, 0x00000000)

    if f.verifyPassword():
        print('Sensor connected successfully!')
    else:
        print('Wrong password!')
except Exception as e:
    print('Error:', e)