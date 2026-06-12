import numpy as np

from utils.frame_transform import apply_camera_rotation, normalize_camera_rotation


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
