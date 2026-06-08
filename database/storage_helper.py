import logging
import os
import tempfile

from firebase_admin import storage

import config.config as config
import utils.utils as utils


logger = logging.getLogger(__name__)


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
