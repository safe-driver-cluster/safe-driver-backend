import queue
import threading

# Shared queue between detect.py and main.py
behavior_queue = queue.Queue()
_latest_camera_frame = None
_latest_camera_frame_lock = threading.Lock()
_behavior_reset_callbacks = []
_behavior_reset_callbacks_lock = threading.Lock()


def update_latest_camera_frame(frame):
    """Store a copy of the latest camera frame for security-alert evidence."""
    global _latest_camera_frame
    with _latest_camera_frame_lock:
        _latest_camera_frame = frame.copy()


def get_latest_camera_frame():
    """Return a safe copy of the latest camera frame."""
    with _latest_camera_frame_lock:
        return None if _latest_camera_frame is None else _latest_camera_frame.copy()


def register_behavior_reset_callback(callback):
    """Register a callback that clears detector behavior state."""
    with _behavior_reset_callbacks_lock:
        if callback not in _behavior_reset_callbacks:
            _behavior_reset_callbacks.append(callback)


def unregister_behavior_reset_callback(callback):
    with _behavior_reset_callbacks_lock:
        if callback in _behavior_reset_callbacks:
            _behavior_reset_callbacks.remove(callback)


def reset_behavior_state(reason="driver_changed"):
    """Request all active detectors to clear driver-specific behavior state."""
    with _behavior_reset_callbacks_lock:
        callbacks = list(_behavior_reset_callbacks)

    for callback in callbacks:
        try:
            callback(reason)
        except Exception:
            pass


stop_event = threading.Event()
