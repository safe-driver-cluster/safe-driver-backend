import numpy as np

import config.config as config
import model.frame_detector as frame_detector


def test_object_detector_frame_payload_carries_current_language(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(config, "LANGUAGE", "SINHALA")

    payload = frame_detector._build_frame_payload(frame)

    assert payload["language"] == "SINHALA"
    assert payload["frame"] is frame


def test_object_detector_frame_payload_falls_back_to_english(monkeypatch):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(config, "LANGUAGE", "UNKNOWN")

    payload = frame_detector._build_frame_payload(frame)

    assert payload["language"] == "ENGLISH"
