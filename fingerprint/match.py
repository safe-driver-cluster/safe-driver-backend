import time
import subprocess
import shutil
import tempfile
import os
import socket
import hashlib
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


def _play_audio_beep():
    """Try to play beep on system audio output (headphones/earbuds)."""
    alsa_sample = Path('/usr/share/sounds/alsa/Front_Center.wav')

    commands = []
    if alsa_sample.exists():
        commands.append(['paplay', str(alsa_sample)])
        commands.append(['aplay', '-q', str(alsa_sample)])

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


def announce(message, speak=True, beep=False):
    print(message)
    if speak:
        _speak_message(message)
    if beep:
        beep_success()


def beep_success():
    if _play_audio_beep():
        return

    buzzer_device = _get_buzzer()
    if buzzer_device is not None:
        buzzer_device.on()
        sleep(0.2)
        buzzer_device.off()


def wait_for_finger(sensor, timeout=10):
    """Wait for finger image read for up to timeout seconds."""
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        if sensor.readImage():
            return True
        time.sleep(0.1)
    return False


def get_scanner_id():
    """Return stable scanner identifier used to build template IDs."""
    env_scanner = os.getenv('SCANNER_ID')
    if env_scanner:
        return env_scanner.strip()

    env_device_mac = os.getenv('DEVICE_MAC')
    if env_device_mac:
        return env_device_mac.strip()

    machine_id_path = Path('/etc/machine-id')
    if machine_id_path.exists():
        try:
            return machine_id_path.read_text(encoding='utf-8').strip()
        except Exception:
            pass

    return socket.gethostname()


def build_fingerprint_template_id(scanner_id, template_position):
    """Create deterministic 32-char ID for scanner template slot."""
    normalized = f"{scanner_id.strip().lower()}:{int(template_position)}"
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def match_fingerprint():
    try:
        sensor = PyFingerprint('/dev/serial0', 57600, 0xFFFFFFFF, 0x00000000)

        if not sensor.verifyPassword():
            announce('Wrong password for fingerprint sensor.')
            return False

        announce('Sensor connected successfully!', speak=False)
        announce('Place finger for verification.')

        if not wait_for_finger(sensor, timeout=10):
            announce('Timeout. Finger not placed.')
            return False

        sensor.convertImage(0x01)
        result = sensor.searchTemplate()
        position = result[0]
        accuracy = result[1]

        if position >= 0:
            scanner_id = get_scanner_id()
            template_fingerprint_id = build_fingerprint_template_id(scanner_id, position)

            # Machine-readable line for integrating with APIs/Firebase update flow.
            print(
                f'FINGERPRINT_MATCH:scanner_id={scanner_id};'
                f'template_position={position};template_id={template_fingerprint_id};accuracy={accuracy}'
            )

            beep_success()
            announce(
                f'Fingerprint matched successfully. Template position {position}. '
                f'Accuracy score {accuracy}. Scanner {scanner_id}. '
                f'Template ID {template_fingerprint_id}.'
            )
            return True

        announce('No fingerprint match found.')
        return False

    except Exception as e:
        announce('Fingerprint matching failed.')
        print('Exception:', e)
        _speak_message(f'Fingerprint matching failed. {e}')
        return False


if __name__ == '__main__':
    success = match_fingerprint()
    if success:
        announce('Fingerprint verification completed successfully.')
    else:
        announce('Fingerprint verification failed or dismissed.')