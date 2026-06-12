import numpy as np

from utils.frame_transform import (
    apply_camera_rotation,
    apply_camera_zoom,
    normalize_camera_rotation,
    normalize_camera_zoom,
)


def test_normalize_camera_rotation_accepts_supported_values():
    assert normalize_camera_rotation(0) == 0
    assert normalize_camera_rotation("90") == 90
    assert normalize_camera_rotation(180) == 180
    assert normalize_camera_rotation(270) == 270
    assert normalize_camera_rotation(450) == 90


def test_normalize_camera_rotation_rejects_unsupported_values():
    assert normalize_camera_rotation(45) == 0
    assert normalize_camera_rotation("upside-down") == 0
    assert normalize_camera_rotation(None) == 0


def test_normalize_camera_zoom_keeps_safe_range():
    assert normalize_camera_zoom(1) == 1.0
    assert normalize_camera_zoom("1.5") == 1.5
    assert normalize_camera_zoom(0.5) == 1.0
    assert normalize_camera_zoom("invalid") == 1.0
    assert normalize_camera_zoom(10) == 4.0


def test_apply_camera_rotation_180():
    frame = np.array(
        [
            [[1, 10, 100], [2, 20, 200]],
            [[3, 30, 300], [4, 40, 400]],
        ],
        dtype=np.uint16,
    )

    rotated = apply_camera_rotation(frame, 180)

    assert rotated.tolist() == [
        [[4, 40, 400], [3, 30, 300]],
        [[2, 20, 200], [1, 10, 100]],
    ]


def test_apply_camera_zoom_preserves_size_and_center_crops():
    frame = np.arange(16, dtype=np.uint8).reshape((4, 4))

    zoomed = apply_camera_zoom(frame, 2)

    assert zoomed.shape == frame.shape
    assert zoomed[0, 0] == frame[1, 1]
    assert zoomed[-1, -1] == frame[2, 2]
