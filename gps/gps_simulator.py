"""
GPS module simulator for SafeDriver.

This script emits real $GPRMC NMEA sentences from the GPS_SIGNALS array below.
Use --pty on Linux/Raspberry Pi to create a virtual serial device that the
backend can read exactly like a GPS module.
"""

import argparse
from datetime import datetime, timezone
import logging
import os
import time

import config.settings as settings


GPS_SIGNALS = [
    # Brief departure sequence, followed by a long period above 20 km/h.
    {"latitude": 7.235866206694628, "longitude": 80.30873860669969, "speed_kmh": 0.0},
    {"latitude": 7.2352210809659345, "longitude": 80.31033758898538, "speed_kmh": 6.0},
    {"latitude": 7.235189234916461, "longitude": 80.31042319341408, "speed_kmh": 12.0},
    {"latitude": 7.23515208118923, "longitude": 80.31052484867315, "speed_kmh": 20.0},
    {"latitude": 7.235104312106877, "longitude": 80.31062115365542, "speed_kmh": 24.0},
    {"latitude": 7.23505654301946, "longitude": 80.3107121083609, "speed_kmh": 27.0},
    {"latitude": 7.235024696958384, "longitude": 80.3108030630664, "speed_kmh": 29.0},
    {"latitude": 7.23496631250722, "longitude": 80.31091541887906, "speed_kmh": 31.0},
    {"latitude": 7.234913235726895, "longitude": 80.31101172386134, "speed_kmh": 32.0},
    {"latitude": 7.234860158940321, "longitude": 80.31111872939721, "speed_kmh": 34.0},
    {"latitude": 7.234828312865377, "longitude": 80.31123643548666, "speed_kmh": 33.0},
    {"latitude": 7.2347752360688045, "longitude": 80.31133809074575, "speed_kmh": 31.0},
    {"latitude": 7.234722159266011, "longitude": 80.311461147112, "speed_kmh": 30.0},
    {"latitude": 7.234685005500334, "longitude": 80.31155745209428, "speed_kmh": 28.0},
    {"latitude": 7.234658467094407, "longitude": 80.31167515818373, "speed_kmh": 27.0},
    {"latitude": 7.234647851731602, "longitude": 80.3117821637196, "speed_kmh": 26.0},
    {"latitude": 7.234631928686917, "longitude": 80.31192127091623, "speed_kmh": 25.0},
    {"latitude": 7.234889892624378, "longitude": 80.31205306108544, "speed_kmh": 28.0},
    {"latitude": 7.234631931302363, "longitude": 80.31212458023884, "speed_kmh": 27.0},
    {"latitude": 7.234621315789207, "longitude": 80.31222088657951, "speed_kmh": 26.0},
    {"latitude": 7.234637239058858, "longitude": 80.31232254327243, "speed_kmh": 25.0},
    {"latitude": 7.23465847008418, "longitude": 80.31239744820407, "speed_kmh": 24.0},
    {"latitude": 7.234695624376087, "longitude": 80.31248840419246, "speed_kmh": 25.0},
    {"latitude": 7.234722163154164, "longitude": 80.31255260841958, "speed_kmh": 26.0},
    {"latitude": 7.234775240705608, "longitude": 80.31261146229441, "speed_kmh": 27.0},
    {"latitude": 7.234817702742269, "longitude": 80.31268636722605, "speed_kmh": 29.0},
    {"latitude": 7.234876088036165, "longitude": 80.31276662250994, "speed_kmh": 31.0},
    {"latitude": 7.234939951715707, "longitude": 80.31287189834303, "speed_kmh": 33.0},
    {"latitude": 7.234993168781508, "longitude": 80.31297918670803, "speed_kmh": 34.0},
    {"latitude": 7.23501977731205, "longitude": 80.31306501740002, "speed_kmh": 32.0},
    {"latitude": 7.2350410641353555, "longitude": 80.31315084809202, "speed_kmh": 30.0},
    {"latitude": 7.236052215733889, "longitude": 80.31526437503916, "speed_kmh": 35.0},
]


logger = logging.getLogger(__name__)


class _PtyWriter:
    """Keep both PTY endpoints open until the backend connects."""

    def __init__(self, master_fd, slave_fd):
        self._master = os.fdopen(master_fd, "wb", buffering=0)
        self._slave_fd = slave_fd

    def write(self, data):
        return self._master.write(data)

    def flush(self):
        self._master.flush()

    def close(self):
        self._master.close()
        os.close(self._slave_fd)


def _nmea_checksum(sentence_body):
    checksum = 0
    for char in sentence_body:
        checksum ^= ord(char)
    return f"{checksum:02X}"


def _format_latitude(value):
    direction = "N" if value >= 0 else "S"
    absolute = abs(value)
    degrees = int(absolute)
    minutes = (absolute - degrees) * 60
    return f"{degrees:02d}{minutes:07.4f}", direction


def _format_longitude(value):
    direction = "E" if value >= 0 else "W"
    absolute = abs(value)
    degrees = int(absolute)
    minutes = (absolute - degrees) * 60
    return f"{degrees:03d}{minutes:07.4f}", direction


def build_rmc_sentence(signal, timestamp=None):
    timestamp = timestamp or datetime.now(timezone.utc)
    latitude, lat_dir = _format_latitude(signal["latitude"])
    longitude, lon_dir = _format_longitude(signal["longitude"])
    speed_knots = float(signal.get("speed_kmh", 0.0)) / 1.852
    status = signal.get("status", "A")

    body = (
        f"GPRMC,{timestamp:%H%M%S}.00,{status},{latitude},{lat_dir},"
        f"{longitude},{lon_dir},{speed_knots:.4f},0.00,{timestamp:%d%m%y},,,A"
    )
    return f"${body}*{_nmea_checksum(body)}"


def _iter_signals(repeat):
    while True:
        for signal in GPS_SIGNALS:
            yield signal
        if not repeat:
            break


def _write_line(writer, line):
    writer.write((line + "\r\n").encode("ascii"))
    writer.flush()


def _open_serial_writer(port, baudrate):
    import serial

    return serial.Serial(port, baudrate=baudrate, timeout=1)


def _open_pty_writer(link_path):
    if settings.SYSTEM == "windows":
        raise RuntimeError("--pty is only available on Linux/macOS")

    import pty

    master_fd, slave_fd = pty.openpty()
    slave_path = os.ttyname(slave_fd)

    if link_path:
        link_path = os.path.abspath(link_path)
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(slave_path, link_path)
        logger.info("Virtual GPS port: %s -> %s", link_path, slave_path)
        logger.info("Start the backend normally with: python run.py")
    else:
        logger.info("Virtual GPS port: %s", slave_path)
        logger.info("Start backend with: GPS_SERIAL_PORT=%s", slave_path)

    return _PtyWriter(master_fd, slave_fd)


def run_simulator(port=None, baudrate=9600, interval=1.0, repeat=False, pty_mode=False, link_path=None, dry_run=False):
    writer = None
    try:
        if dry_run:
            logger.info("Dry run enabled; printing NMEA sentences only")
        elif pty_mode:
            writer = _open_pty_writer(link_path)
        elif port:
            writer = _open_serial_writer(port, baudrate)
            logger.info("Writing GPS signals to %s at %s baud", port, baudrate)
        else:
            raise ValueError("Choose --pty, --port, or --dry-run")

        for index, signal in enumerate(_iter_signals(repeat), start=1):
            sentence = build_rmc_sentence(signal)
            logger.info(
                "GPS signal %s: speed=%.2f km/h lat=%.6f lon=%.6f nmea=%s",
                index,
                float(signal.get("speed_kmh", 0.0)),
                signal["latitude"],
                signal["longitude"],
                sentence,
            )

            if writer is not None:
                _write_line(writer, sentence)

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("GPS simulator stopped by user")
    finally:
        if writer is not None:
            writer.close()
        if pty_mode and link_path and os.path.islink(link_path):
            os.remove(link_path)


def parse_args():
    parser = argparse.ArgumentParser(description="SafeDriver GPS NMEA signal simulator")
    parser.add_argument("--port", help="Serial/virtual serial port to write to")
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between signals")
    parser.add_argument("--repeat", action="store_true", help="Repeat GPS_SIGNALS forever")
    parser.add_argument("--pty", action="store_true", help="Create a virtual GPS serial port")
    parser.add_argument(
        "--link",
        default="/tmp/safe_driver_gps",
        help="Stable symlink path for --pty mode",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print generated NMEA")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    run_simulator(
        port=args.port,
        baudrate=args.baudrate,
        interval=args.interval,
        repeat=args.repeat,
        pty_mode=args.pty,
        link_path=args.link,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
