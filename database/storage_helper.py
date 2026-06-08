import logging
import os
import tempfile
import time
from urllib.parse import quote
import uuid

from firebase_admin import storage

import config.config as config
import utils.utils as utils


logger = logging.getLogger(__name__)


def upload_alert_evidence(image_bytes, device_mac, event_type):
    """Upload alert evidence JPEG and return its Firebase download URL."""
    if not image_bytes:
        return None

    safe_mac = (device_mac or "unknown-device").replace(":", "-")
    safe_event = "".join(
        char if char.isalnum() or char in ("-", "_") else "-"
        for char in (event_type or "alert")
    )
    token = str(uuid.uuid4())
    object_name = (
        f"{config.ALERT_EVIDENCE_STORAGE_PREFIX}/{safe_mac}/"
        f"{safe_event}-{int(time.time() * 1000)}-{token}.jpg"
    )

    bucket = storage.bucket(config.FIREBASE_STORAGE_BUCKET)
    blob = bucket.blob(object_name)
    blob.metadata = {"firebaseStorageDownloadTokens": token}
    blob.upload_from_string(image_bytes, content_type="image/jpeg")

    encoded_name = quote(object_name, safe="")
    url = (
        f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/"
        f"{encoded_name}?alt=media&token={token}"
    )
    logger.info("Alert evidence uploaded: %s", object_name)
    return {"url": url, "path": object_name}


def sync_voice_alerts_from_storage():
    """Download Firebase voice-alert folders and replace local files."""
    bucket = storage.bucket(config.FIREBASE_STORAGE_BUCKET)
    audio_root = utils.get_audio_dir()
    downloaded = 0
    failed = 0

    for remote_language, local_language in config.VOICE_ALERT_LANGUAGES.items():
        prefix = f"{config.VOICE_ALERT_STORAGE_PREFIX}/{remote_language}/"
        local_language_dir = os.path.join(audio_root, local_language)
        os.makedirs(local_language_dir, exist_ok=True)

        for blob in bucket.list_blobs(prefix=prefix):
            relative_name = blob.name[len(prefix):]
            if not relative_name or relative_name.endswith("/"):
                continue

            # Storage object paths must not escape the local language folder.
            relative_name = relative_name.replace("\\", "/")
            destination = os.path.abspath(os.path.join(local_language_dir, *relative_name.split("/")))
            language_root = os.path.abspath(local_language_dir)
            if os.path.commonpath([language_root, destination]) != language_root:
                logger.warning("Skipped unsafe voice-alert object path: %s", blob.name)
                failed += 1
                continue

            os.makedirs(os.path.dirname(destination), exist_ok=True)
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    dir=os.path.dirname(destination),
                    prefix=".voice-alert-",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temp_path = temp_file.name

                blob.download_to_filename(temp_path)
                os.replace(temp_path, destination)
                downloaded += 1
            except Exception:
                failed += 1
                logger.exception("Failed to download voice-alert object: %s", blob.name)
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

    logger.info(
        "Voice-alert sync completed: %s downloaded/replaced, %s failed, destination=%s",
        downloaded,
        failed,
        audio_root,
    )
    return {
        "success": failed == 0,
        "downloaded": downloaded,
        "failed": failed,
        "destination": audio_root,
    }
