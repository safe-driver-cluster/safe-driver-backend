import time
import subprocess
import shutil
import tempfile
import os
import socket
import hashlib
import logging
from pathlib import Path
from time import sleep
from database.firestore_helper import FirestoreHelper
from database import db_helper
from service.model_service import (get_mac_address_alternative)
import config.config as config
from fingerprint.sensor import close_sensor, create_sensor, is_packet_header_error
from service.driver_auth_service import driver_auth_service
from shared import stop_event

firestore_helper = FirestoreHelper()
logger = logging.getLogger(__name__)

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


def wait_for_finger(sensor, timeout=10):
    """Wait for finger image read"""
    while True:
        if sensor.readImage():
            return True
        time.sleep(0.1)
    return False


def wait_for_finger_removal(sensor):
    """Prevent one physical finger placement from counting multiple times."""
    while not stop_event.is_set():
        try:
            if not sensor.readImage():
                return
        except Exception as exc:
            logger.warning("Fingerprint read failed while waiting for removal: %s", exc)
            return
        time.sleep(0.1)


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
    sensor = None
    try:
        sensor = create_sensor()
        packet_error_count = 0

        while not stop_event.is_set():
            if not driver_auth_service.is_verified():
                driver_auth_service.request_verification_if_due()

            try:
                has_image = sensor.readImage()
            except Exception as exc:
                if not is_packet_header_error(exc):
                    raise

                packet_error_count += 1
                sensor_port = getattr(sensor, "safe_driver_port", "unknown")
                logger.warning(
                    (
                        "Fingerprint serial packet error on %s (%s/%s): %s. "
                        "This usually means serial console noise, wrong UART port, wrong baudrate, "
                        "or another process is reading the sensor."
                    ),
                    sensor_port,
                    packet_error_count,
                    config.FINGERPRINT_PACKET_ERROR_REOPEN_THRESHOLD,
                    exc,
                )

                if packet_error_count >= config.FINGERPRINT_PACKET_ERROR_REOPEN_THRESHOLD:
                    close_sensor(sensor)
                    stop_event.wait(config.FINGERPRINT_SENSOR_REOPEN_DELAY_SEC)
                    try:
                        sensor = create_sensor(exclude_ports={sensor_port})
                    except Exception:
                        logger.exception("Fingerprint sensor reopen on alternate port failed; retrying primary list")
                        sensor = create_sensor()
                    packet_error_count = 0

                time.sleep(0.1)
                continue

            packet_error_count = 0

            if not has_image:
                time.sleep(0.1)
                continue

            try:
                sensor.convertImage(0x01)
                result = sensor.searchTemplate()
            except Exception as exc:
                if is_packet_header_error(exc):
                    logger.warning("Fingerprint packet error during template search: %s", exc)
                    close_sensor(sensor)
                    stop_event.wait(config.FINGERPRINT_SENSOR_REOPEN_DELAY_SEC)
                    sensor = create_sensor()
                    continue
                raise

            position = result[0]
            accuracy = result[1]

            if position >= 0:
                scanner_id = get_scanner_id()
                template_fingerprint_id = build_fingerprint_template_id(scanner_id, position)

                # Machine-readable line for integrating with APIs/Firebase update flow.
                logger.info(
                    f'FINGERPRINT_MATCH: scanner_id= {scanner_id} | template_position= {position} | accuracy= {accuracy} | template_id= {template_fingerprint_id}'
                )

                driver_id = firestore_helper.get_driver_by_fingerprint(scanner_id=scanner_id, template_position=position)
                if driver_id:
                    current_driver_id = driver_auth_service.verified_driver()
                    device_mac = get_mac_address_alternative().upper()

                    if current_driver_id:
                        if driver_id == current_driver_id:
                            db_helper.update_assigned_driver(None, device_mac)
                            db_helper.update_device_verification(device_mac, False)
                            driver_auth_service.mark_signed_off()
                            firestore_helper.record_driver_fingerprint_operation(
                                driver_id=driver_id,
                                device_mac=device_mac,
                                scanner_id=scanner_id,
                                template_position=position,
                                accuracy=accuracy,
                                operation="driver_sign_off",
                            )
                            beep_success()
                            logger.info("Registered driver signed off: driver_id=%s", driver_id)
                        else:
                            driver_auth_service.warn_wrong_signoff_driver()
                            logger.warning(
                                "Fingerprint sign-off rejected: verified_driver=%s scanned_driver=%s",
                                current_driver_id,
                                driver_id,
                            )
                        wait_for_finger_removal(sensor)
                        continue

                    driver_obj = firestore_helper.get_driver(driver_id) or {}
                    driver_name = driver_obj.get('name', 'Unknown')
                    driver_language = driver_obj.get('language', 'ENGLISH')

                    db_helper.update_assigned_driver(driver_id, device_mac)
                    db_helper.update_device_verification(device_mac, True)
                    driver_auth_service.mark_verified(driver_id, driver_language)
                    firestore_helper.record_driver_fingerprint_operation(
                        driver_id=driver_id,
                        device_mac=device_mac,
                        scanner_id=scanner_id,
                        template_position=position,
                        accuracy=accuracy,
                        operation="driver_verification",
                    )
                    beep_success()
                    logger.info(
                        "Registered driver authenticated: driver_id=%s name=%s",
                        driver_id,
                        driver_name,
                    )
                    wait_for_finger_removal(sensor)
                    continue
                    
                else:
                    driver_auth_service.record_unauthorized_attempt()
                    wait_for_finger_removal(sensor)

            else:
                driver_auth_service.record_unauthorized_attempt()
                wait_for_finger_removal(sensor)
            
        return True

    except Exception as e:
        logger.exception("Fingerprint matching failed")
        return False
    finally:
        if sensor is not None:
            close_sensor(sensor)


def main():
    while not stop_event.is_set():
        if match_fingerprint():
            break
        logger.warning("Fingerprint verification worker retrying in 5 seconds")
        stop_event.wait(5)

    logger.info(
        "Fingerprint verification worker stopped: verified=%s",
        driver_auth_service.is_verified(),
    )

if __name__ == '__main__':
    main()
