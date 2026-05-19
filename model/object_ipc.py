import os
import sys
import struct
import subprocess
import threading
import queue
import logging
from typing import Optional, Callable

import cv2

logger = logging.getLogger(__name__)

class ObjectDetectClient:
    def __init__(
        self,
        *,
        python_exe: str,
        cwd: str,
        on_line: Optional[Callable[[str], None]] = None,
        max_queue: int = 2,
        jpeg_quality: int = 75,
    ):
        self._python_exe = python_exe
        self._cwd = cwd
        self._on_line = on_line
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=max_queue)
        self._proc: Optional[subprocess.Popen] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._jpeg_quality = int(jpeg_quality)

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return

        self._proc = subprocess.Popen(
            [self._python_exe, "-u", "-c", "from model.object_detect import main; main()"],
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            universal_newlines=True,
        )

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def stop(self) -> None:
        if not self._proc:
            return
        try:
            self._proc.terminate()
        except Exception:
            pass

    def is_running(self) -> bool:
        return bool(self._proc and self._proc.poll() is None)

    def try_send_frame(self, frame_bgr) -> bool:
        if not self.is_running():
            return False

        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            return False

        payload = buf.tobytes()

        try:
            self._q.put_nowait(payload)
            return True
        except queue.Full:
            return False  # drop frame

    def _writer_loop(self) -> None:
        assert self._proc and self._proc.stdin
        w = self._proc.stdin
        while self.is_running():
            payload = self._q.get()
            try:
                w.write(struct.pack(">I", len(payload)))
                w.write(payload)
                w.flush()
            except Exception as exc:
                logger.warning("Object worker write failed: %s", exc)
                break

    def _reader_loop(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            if self._on_line:
                try:
                    self._on_line(line)
                except Exception:
                    logger.exception("Object worker line handler failed")
