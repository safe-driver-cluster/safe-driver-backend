import time
import subprocess
import shutil
import tempfile
import logging
from pathlib import Path
from fingerprint.operation import exclusive_fingerprint_operation
from fingerprint.sensor import close_sensor, create_sensor
from time import sleep
from database.firestore_helper import FirestoreHelper

try:
    from gtts import gTTS
except Exception:
    gTTS = None

buzzer = None
buzzer_checked = False

firestore_helper = FirestoreHelper()
logger = logging.getLogger(__name__)

def _get_buzzer():
    """Initialize GPIO buzzer only when needed."""
    global buzzer, buzzer_checked

    if buzzer_checked:
        return buzzer

    buzzer_checked = True

    try:
        from gpiozero import Buzzer
        buzzer = Buzzer(17)  # Optional GPIO buzzer on pin 17
    except Exception:
        buzzer = None

    return buzzer


def _play_audio_beep():
    """Try to play beep on system audio output (headphones/earbuds)."""
    alsa_sample = Path('/usr/share/sounds/alsa/Front_Center.wav')

    commands = []
    if alsa_sample.exists():
        commands.append(['paplay', str(alsa_sample)])
        commands.append(['aplay', '-q', str(alsa_sample)])

    # Fallback tone if sample file is not available.
    commands.append(['speaker-test', '-q', '-t', 'sine', '-f', '1000', '-l', '1'])

    for command in commands:
        if shutil.which(command[0]) is None:
            continue

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
            if result.returncode == 0:
                return True
        except Exception:
            continue

    return False


def _speak_message(message):
    """Speak a short message through the current audio output device."""
    # Primary path: gTTS (same speech generation mechanism) + CLI player.
    if gTTS is not None:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name

            tts = gTTS(text=message, lang='en')
            tts.save(temp_path)
            if _play_audio_file(temp_path):
                return True
        except Exception:
            pass
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    # Fallback path: local TTS binaries if available.
    tts_commands = [
        ['spd-say', '-w', message],
        ['espeak', message],
    ]

    for command in tts_commands:
        if shutil.which(command[0]) is None:
            continue

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                check=False,
            )
            if result.returncode == 0:
                return True
        except Exception:
            continue

    return False


def _play_audio_file(file_path):
    """Play audio file using available command-line players."""
    commands = [
        ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', file_path],
        ['mpg123', '-q', file_path],
        ['mpg321', '-q', file_path],
    ]

    for command in commands:
        if shutil.which(command[0]) is None:
            continue

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=12,
                check=False,
            )
            if result.returncode == 0:
                return True
        except Exception:
            continue

    return False


def announce(message, speak=True, beep=False):
    logger.info(message)
    if speak:
        _speak_message(message)
    if beep:
        beep_success()

def beep_success():
    # Prefer audio so beeps can be heard on earbuds/headphones.
    if _play_audio_beep():
        return

    # Fallback to hardware GPIO buzzer if present.
    buzzer_device = _get_buzzer()
    if buzzer_device is not None:
        buzzer_device.on()
        sleep(0.2)
        buzzer_device.off()

# -------------------------------
# Function to wait for finger
# -------------------------------
def wait_for_finger(sensor, timeout=5):
    """Wait for finger for `timeout` seconds."""
    if sensor is None:
        announce('Fingerprint sensor is not initialized.')
        return False

    start_time = time.time()
    while (time.time() - start_time) < timeout:
        if sensor.readImage():
            return True
        time.sleep(0.1)
    return False

# -------------------------------
# Function to enroll fingerprint
# -------------------------------
def enroll_fingerprint():
    sensor = None
    with exclusive_fingerprint_operation("enroll"):
        try:
            sensor = create_sensor()
            announce('Sensor connected successfully!', speak=False)
            return _enroll_fingerprint(sensor)
        except Exception as e:
            announce('Sensor initialization failed!')
            print('Exception:', e)
            _speak_message(f'Sensor initialization failed. {e}')
            return False
        finally:
            if sensor is not None:
                close_sensor(sensor)


def _enroll_fingerprint(sensor):
    announce('Starting fingerprint enrollment...')

    # Step 1: Turn LED blue (ready)
    announce('Ready for first scan.')
    # If LED color command exists, implement here
    
    # Step 2: Wait for first finger
    announce('Place finger for first scan.')
    if not wait_for_finger(sensor, 5):
        announce('Timeout. Finger not placed.')
        return False

    # Step 3: Convert image to characteristics
    sensor.convertImage(0x01)
    beep_success()
    announce('First scan successful!')

    # Step 4: Ask for second scan
    announce('Please place the same finger again for second scan.')
    # Step 5: Turn LED Orange
    announce('Second scan in progress.', speak=False)
    # If LED color command exists, implement here

    if not wait_for_finger(sensor, 5):
        announce('Timeout. Finger not placed.')
        return False

    # Step 6: Convert image to characteristics
    sensor.convertImage(0x02)

    # Step 7: Compare characteristics
    if sensor.compareCharacteristics() == 0:
        announce('Fingerprints do not match. Operation dismissed.')
        return False

    # Step 8: Create template
    positionNumber = sensor.storeTemplate()
    announce(f'Second scan successful. Fingerprint enrolled successfully! Template position: {positionNumber}')
    beep_success()

    # Step 9: Turn LED back to Blue
    announce('Enrollment complete.')
    # If LED color command exists, implement here

    return True

def enroll_fingerprint_with_id(driver_id: str):
    payload = None
    sensor = None
    with exclusive_fingerprint_operation("enroll_with_id"):
        try:
            sensor = create_sensor()
            announce('Sensor connected successfully!', speak=False)
            return _enroll_fingerprint_with_id(sensor, driver_id)
        except Exception as e:
            announce('Sensor initialization failed!')
            print('Exception:', e)
            _speak_message(f'Sensor initialization failed. {e}')
            return payload
        finally:
            if sensor is not None:
                close_sensor(sensor)


def _enroll_fingerprint_with_id(sensor, driver_id: str):
    payload = None

    announce('Starting fingerprint enrollment...')

    # Step 1: Turn LED blue (ready)
    announce('Ready for first scan.')
    # If LED color command exists, implement here
    
    # Step 2: Wait for first finger
    announce('Place finger for first scan.')
    if not wait_for_finger(sensor, 10):
        announce('Timeout. Finger not placed.')
        return payload

    # Step 3: Convert image to characteristics
    sensor.convertImage(0x01)
    beep_success()
    announce('First scan successful!')

    # Step 4: Ask for second scan
    announce('Please place the same finger again for second scan.')
    # Step 5: Turn LED Orange
    announce('Second scan in progress.', speak=False)
    # If LED color command exists, implement here

    if not wait_for_finger(sensor, 10):
        announce('Timeout. Finger not placed.')
        return payload

    # Step 6: Convert image to characteristics
    sensor.convertImage(0x02)

    # Step 7: Compare characteristics
    if sensor.compareCharacteristics() == 0:
        announce('Fingerprints do not match. Operation dismissed.')
        return payload

    # Step 8: Create template
    positionNumber = sensor.storeTemplate()
    announce(f'Second scan successful. Fingerprint enrolled successfully! Template position: {positionNumber}')
    beep_success()

    # Step 9: Save template_id to firestore
    payload = firestore_helper.register_driver_fingerprint(driver_id=driver_id, template_position=positionNumber)

    # Step 10: Turn LED back to Blue
    announce('Enrollment complete.')
    # If LED color command exists, implement here

    return payload

# -------------------------------
# Execute enrollment
# -------------------------------
if __name__ == '__main__':
    success = enroll_fingerprint()
    if success:
        announce('Enrollment finished successfully.')
    else:
        announce('Enrollment failed or dismissed.')
