import serial
import pynmea2
import logging

# Ignore speeds below 2 km/h when stationary
SPEED_THRESHOLD = 2.0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

try:
    gps = serial.Serial("/dev/ttyAMA5", baudrate=9600, timeout=1)
    logger.info("Serial port opened successfully")
except serial.SerialException as e:
    logger.error(f"Failed to open serial port: {e}")
    raise

logger.info("Reading GPS...")

while True:
    try:
        raw = gps.readline()
        line = raw.decode('utf-8', errors='ignore').strip()

        if not line:
            continue

        if line.startswith(("$GPRMC", "$GNRMC")):
            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError as e:
                logger.warning(f"Parse error: {e}")
                continue

            if msg.status != 'A':
                logger.warning("No GPS fix")
                continue

            lat = msg.latitude
            lng = msg.longitude
            
            speed_kmh = float(msg.spd_over_grnd) * 1.852 if msg.spd_over_grnd else 0.0

            # Filter out jitter
            if speed_kmh < SPEED_THRESHOLD:
                speed_kmh = 0.0

            logger.info(f"Latitude : {lat}")
            logger.info(f"Longitude: {lng}")
            logger.info(f"Speed    : {round(speed_kmh, 2)} km/h")
            logger.info("----------------------")

    except serial.SerialException as e:
        logger.error(f"Serial error: {e}")
        break
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")