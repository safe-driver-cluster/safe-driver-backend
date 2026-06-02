import serial
import pynmea2
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # gps = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)
    gps = serial.Serial("/dev/ttyAMA4", baudrate=9600, timeout=1)
    logger.info("Serial port opened successfully")
except serial.SerialException as e:
    logger.error(f"Failed to open serial port: {e}")
    raise

logger.info("Reading GPS... (blue blinking = searching for satellites, wait 1-5 min outdoors)")

while True:
    try:
        raw = gps.readline()
        line = raw.decode('utf-8', errors='ignore').strip()

        if not line:
            continue

        # Log ALL sentences at DEBUG level so you can see data is arriving
        logger.debug(f"RAW: {line}")

        # Handle both GPRMC (GPS only) and GNRMC (GPS + GLONASS)
        if line.startswith(("$GPRMC", "$GNRMC")):
            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError as e:
                logger.warning(f"Parse error: {e} | Line: {line}")
                continue

            # 'A' = Active fix, 'V' = Void (no fix yet)
            if msg.status != 'A':
                logger.warning("No GPS fix yet (status=V) — go outdoors or wait longer")
                continue

            lat = msg.latitude
            lng = msg.longitude
            speed_knots = msg.spd_over_grnd
            speed_kmh = float(speed_knots) * 1.852 if speed_knots else 0.0

            logger.info(f"Latitude : {lat}")
            logger.info(f"Longitude: {lng}")
            logger.info(f"Speed    : {round(speed_kmh, 2)} km/h")
            logger.info("----------------------")

    except serial.SerialException as e:
        logger.error(f"Serial error: {e}")
        break
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")