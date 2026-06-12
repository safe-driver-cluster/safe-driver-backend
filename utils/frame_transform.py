import cv2


def normalize_camera_rotation(rotation):
    """Return a supported clockwise camera rotation angle."""
    try:
        normalized = int(rotation) % 360
    except (TypeError, ValueError):
        return 0

    if normalized in (0, 90, 180, 270):
        return normalized
    return 0


def apply_camera_rotation(frame, rotation):
    """Rotate a camera frame by a supported clockwise angle."""
    rotation = normalize_camera_rotation(rotation)
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame
