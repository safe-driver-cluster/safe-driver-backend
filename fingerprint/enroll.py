import time
import subprocess
import shutil
import tempfile
from pathlib import Path
from pyfingerprint.pyfingerprint import PyFingerprint
from time import sleep

try:
    from gtts import gTTS
except Exception:
    gTTS = None

buzzer = None
buzzer_checked = False


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
    print(message)
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
# Initialize Fingerprint Sensor
# -------------------------------
try:
    f = PyFingerprint('/dev/serial0', 57600, 0xFFFFFFFF, 0x00000000)

    if f.verifyPassword():
        announce('Sensor connected successfully!', speak=False)
    else:
        raise Exception('Wrong password!')

except Exception as e:
    announce('Sensor initialization failed!')
    print('Exception:', e)
    _speak_message(f'Sensor initialization failed. {e}')
    exit(1)

# -------------------------------
# Function to wait for finger
# -------------------------------
def wait_for_finger(timeout=5):
    """Wait for finger for `timeout` seconds."""
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        if f.readImage():
            return True
        time.sleep(0.1)
    return False

# -------------------------------
# Function to enroll fingerprint
# -------------------------------
def enroll_fingerprint():
    announce('Starting fingerprint enrollment...')

    # Step 1: Turn LED blue (ready)
    announce('Ready for first scan.')
    # If LED color command exists, implement here
    
    # Step 2: Wait for first finger
    announce('Place finger for first scan.')
    if not wait_for_finger(5):
        announce('Timeout. Finger not placed.')
        return False

    # Step 3: Convert image to characteristics
    f.convertImage(0x01)
    beep_success()
    announce('First scan successful!')

    # Step 4: Ask for second scan
    announce('Please place the same finger again for second scan.')
    # Step 5: Turn LED Orange
    announce('Second scan in progress.', speak=False)
    # If LED color command exists, implement here

    if not wait_for_finger(5):
        announce('Timeout. Finger not placed.')
        return False

    # Step 6: Convert image to characteristics
    f.convertImage(0x02)

    # Step 7: Compare characteristics
    if f.compareCharacteristics() == 0:
        announce('Fingerprints do not match. Operation dismissed.')
        return False

    # Step 8: Create template
    positionNumber = f.storeTemplate()
    announce(f'Second scan successful. Fingerprint enrolled successfully! Template position: {positionNumber}')
    beep_success()

    # Step 9: Turn LED back to Blue
    announce('Enrollment complete.')
    # If LED color command exists, implement here

    return True

# -------------------------------
# Execute enrollment
# -------------------------------
if __name__ == '__main__':
    success = enroll_fingerprint()
    if success:
        announce('Enrollment finished successfully.')
    else:
        announce('Enrollment failed or dismissed.')