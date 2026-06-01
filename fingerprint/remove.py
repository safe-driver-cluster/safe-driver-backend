from pyfingerprint.pyfingerprint import PyFingerprint

def delete_fingerprint(position):
    try:
        sensor = PyFingerprint('/dev/serial0', 57600, 0xFFFFFFFF, 0x00000000)

        if sensor.deleteTemplate(position):
            print(f'Fingerprint at position {position} deleted successfully.')
            return True
        else:
            print('Delete failed.')
            return False

    except Exception as e:
        print('Exception:', e)
        return False

if __name__ == '__main__':
    success = delete_fingerprint()
    if success:
        announce('Fingerprint deleted successfully.')
    else:
        announce('Fingerprint deletion failed or dismissed.')