import logging
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import config.config as config


logger = logging.getLogger(__name__)


def _run_audio_command(command):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def set_system_volume(volume_percent, reason="audio"):
    """Set the default speaker/Bluetooth output volume."""
    volume_percent = max(0, min(100, int(volume_percent)))
    volume_ratio = f"{volume_percent / 100:.2f}"
    volume_percent_text = f"{volume_percent}%"
    commands = []

    if shutil.which("pactl") is not None:
        commands.extend(
            [
                ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"],
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", volume_percent_text],
            ]
        )

    if shutil.which("wpctl") is not None:
        commands.extend(
            [
                ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"],
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", volume_ratio],
            ]
        )

    if shutil.which("amixer") is not None:
        commands.extend(
            [
                ["amixer", "-q", "sset", "Master", volume_percent_text, "unmute"],
                ["amixer", "-q", "sset", "PCM", volume_percent_text, "unmute"],
            ]
        )

    if not commands:
        logger.debug("No system audio volume command is available")
        return

    volume_changed = False
    for command in commands:
        volume_changed = _run_audio_command(command) or volume_changed

    if volume_changed:
        logger.info("%s volume set to %s", reason, volume_percent_text)
    else:
        logger.debug("Could not adjust %s volume", reason)


def _set_system_volume_max():
    """Raise the default speaker/Bluetooth output volume before the beep."""
    if not config.SPEAKER_BEEP_AUTO_MAX_VOLUME:
        return

    set_system_volume(config.SPEAKER_BEEP_VOLUME_PERCENT, reason="Speaker beep")


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


def _play_with_pygame(file_path):
    """Play through pygame so Bluetooth follows the same path as voice alerts."""
    try:
        # Let SDL choose the default audio device; this is usually the active
        # Bluetooth sink when voice alerts are already audible there.
        os.environ.setdefault("SDL_AUDIODRIVER", config.SPEAKER_BEEP_SDL_AUDIO_DRIVER)

        import pygame

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        sound = pygame.mixer.Sound(str(file_path))
        sound.set_volume(max(0.0, min(1.0, config.SPEAKER_BEEP_VOLUME)))
        channel = sound.play()
        if channel is None:
            return False

        started_at = time.time()
        while channel.get_busy():
            if time.time() - started_at >= config.SPEAKER_BEEP_TIMEOUT_SEC:
                channel.stop()
                return False
            pygame.time.Clock().tick(20)

        logger.info("Speaker beep fallback played using pygame")
        return True
    except Exception as exc:
        logger.warning("pygame speaker beep failed: %s", exc)
        return False


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
        _set_system_volume_max()

        if config.SPEAKER_BEEP_USE_PYGAME and _play_with_pygame(temp_path):
            return True

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
