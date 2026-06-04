import serial
import pynmea2
import logging
import time
from database.firestore_helper import FirestoreHelper
from database import db_helper
firestore_helper = FirestoreHelper()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)

gps = serial.Serial("/dev/ttyAMA5", baudrate=9600, timeout=1)
logger.info("GPS started, connecting to Firebase...")

last_push_time = 0
last_speed = 0.0

while True:
    try:
        raw = gps.readline()
        line = raw.decode('utf-8', errors='ignore').strip()

        if not line:
            continue

        if line.startswith(("$GPRMC", "$GNRMC")):
            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError:
                continue

            if msg.status != 'A':
                continue

            lat = msg.latitude
            lng = msg.longitude
            speed_kmh = float(msg.spd_over_grnd) * 1.852 if msg.spd_over_grnd else 0.0

            if speed_kmh < SPEED_THRESHOLD:
                speed_kmh = 0.0

            is_overspeeding = speed_kmh > SPEED_LIMIT
            now = time.time()

            # Push if: 5 seconds passed OR speed changed significantly
            speed_changed = abs(speed_kmh - last_speed) >= SPEED_CHANGE_THRESHOLD
            time_elapsed = (now - last_push_time) >= PUSH_INTERVAL

            if time_elapsed or speed_changed:
                data = {
                    "lat": round(lat, 6),
                    "lng": round(lng, 6),
                    "speed_kmh": round(speed_kmh, 2),
                    "is_overspeeding": is_overspeeding,
                    "timestamp": int(now)
                }

                ref.update(data)

                last_push_time = now
                last_speed = speed_kmh

                if is_overspeeding:
                    logger.warning(f"⚠️  OVERSPEED: {round(speed_kmh, 2)} km/h")
                else:
                    logger.info(f"✅ Speed: {round(speed_kmh, 2)} km/h | Pushed to Firebase")

    except serial.SerialException as e:
        logger.error(f"Serial error: {e}")
        break
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")