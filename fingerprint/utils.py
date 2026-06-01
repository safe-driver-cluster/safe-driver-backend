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