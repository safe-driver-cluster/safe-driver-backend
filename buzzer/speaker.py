import logging
import math
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

import config.config as config


logger = logging.getLogger(__name__)


def _create_beep_wav(file_path):
    """Create the configured high-frequency speaker alert pattern."""
    sample_rate = 44100
    amplitude = int(32767 * max(0.0, min(1.0, config.SPEAKER_BEEP_VOLUME)))

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for on_duration, off_duration in config.BUZZER_ALERT_PATTERN:
            on_samples = int(sample_rate * max(0.0, float(on_duration)))
            off_samples = int(sample_rate * max(0.0, float(off_duration)))
            frames = bytearray()

            for index in range(on_samples):
                value = int(
                    amplitude
                    * math.sin(
                        2.0
                        * math.pi
                        * config.SPEAKER_BEEP_FREQUENCY_HZ
                        * index
                        / sample_rate
                    )
                )
                frames.extend(struct.pack("<h", value))

            if off_samples:
                frames.extend(b"\x00\x00" * off_samples)

            wav_file.writeframes(frames)


def speaker_beep():
    """Play a high-frequency alert through the system speaker."""
    if not config.ENABLE_SPEAKER_BEEP_FALLBACK:
        return False

    player_commands = (
        ("aplay", ["aplay", "-q"]),
        ("paplay", ["paplay"]),
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]),
    )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        _create_beep_wav(temp_path)

        for executable, command in player_commands:
            if shutil.which(executable) is None:
                continue

            result = subprocess.run(
                [*command, str(temp_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=config.SPEAKER_BEEP_TIMEOUT_SEC,
                check=False,
            )
            if result.returncode == 0:
                logger.info("Speaker beep fallback played using %s", executable)
                return True

        logger.warning("Speaker beep fallback unavailable: no working audio player found")
    except Exception as exc:
        logger.warning("Speaker beep fallback failed: %s", exc)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return False
