import queue
import threading

# Shared queue between detect.py and main.py
behavior_queue = queue.Queue()
_latest_camera_frame = None
_latest_camera_frame_lock = threading.Lock()


def update_latest_camera_frame(frame):
    """Store a copy of the latest camera frame for security-alert evidence."""
    global _latest_camera_frame
    with _latest_camera_frame_lock:
        _latest_camera_frame = frame.copy()


def get_latest_camera_frame():
    """Return a safe copy of the latest camera frame."""
    with _latest_camera_frame_lock:
        return None if _latest_camera_frame is None else _latest_camera_frame.copy()


stop_event = threading.Event()
