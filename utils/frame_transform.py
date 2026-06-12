import cv2


MAX_CAMERA_ZOOM = 4.0


def normalize_camera_rotation(rotation):
    """Return a supported clockwise camera rotation angle."""
    try:
        normalized = int(rotation) % 360
    except (TypeError, ValueError):
        return 0

    if normalized in (0, 90, 180, 270):
        return normalized
    return 0


def normalize_camera_zoom(zoom):
    """Return a safe digital zoom factor."""
    try:
        normalized = float(zoom)
    except (TypeError, ValueError):
        return 1.0

    if normalized < 1.0:
        return 1.0
    if normalized > MAX_CAMERA_ZOOM:
        return MAX_CAMERA_ZOOM
    return normalized


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


def apply_camera_zoom(frame, zoom):
    """Apply centered digital zoom while preserving the original frame size."""
    zoom = normalize_camera_zoom(zoom)
    if zoom <= 1.0:
        return frame

    height, width = frame.shape[:2]
    crop_width = max(1, int(width / zoom))
    crop_height = max(1, int(height / zoom))

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = frame[top:top + crop_height, left:left + crop_width]

    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
