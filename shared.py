import queue
import threading

# Shared queue between detect.py and main.py
behavior_queue = queue.Queue()
stop_event = threading.Event()  # ← add this