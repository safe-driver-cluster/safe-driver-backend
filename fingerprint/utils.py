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
import logging

try:
    from gtts import gTTS
except Exception:
    gTTS = None

logger = logging.getLogger(__name__)

_AUDIO_PLAYER = None

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

def wait_for_finger(sensor, timeout=10):
    """Wait for finger image read"""
    while True:
        if sensor.readImage():
            return True
        time.sleep(0.1)
    return False

def init_audio():
    global _AUDIO_PLAYER

    for player in ['ffplay', 'mpg123', 'mpg321']:
        if shutil.which(player):
            _AUDIO_PLAYER = player
            break

def _play_audio_file(file_path):
    if not _AUDIO_PLAYER:
        return False

    if _AUDIO_PLAYER == 'ffplay':
        cmd = [
            'ffplay',
            '-nodisp',
            '-autoexit',
            '-loglevel',
            'quiet',
            file_path
        ]
    else:
        cmd = [_AUDIO_PLAYER, '-q', file_path]

    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
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
    logger.info(message)
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