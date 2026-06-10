import os

import model.utilmethods as utilmethods


def test_hazard_voice_filename_uses_label_and_language(monkeypatch, tmp_path):
    monkeypatch.setattr(utilmethods.utils, "get_audio_dir", lambda: str(tmp_path))

    filename = utilmethods.get_voice_alert_file(
        "VOICE_ALERT_HAZARD_DANGEROUS_BEND",
        "ENGLISH",
    )

    assert filename == os.path.join(
        str(tmp_path),
        "ENGLISH",
        "VOICE_ALERT_HAZARD_DANGEROUS_BEND_ENGLISH.mp3",
    )


def test_unknown_language_falls_back_to_english(monkeypatch, tmp_path):
    monkeypatch.setattr(utilmethods.utils, "get_audio_dir", lambda: str(tmp_path))

    filename = utilmethods.get_voice_alert_file(
        "VOICE_ALERT_HAZARD_SLIPPERY_ROAD",
        "UNKNOWN",
    )

    assert filename.endswith(
        os.path.join(
            "ENGLISH",
            "VOICE_ALERT_HAZARD_SLIPPERY_ROAD_ENGLISH.mp3",
        )
    )
