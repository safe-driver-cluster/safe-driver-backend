import serial
import pynmea2
import logging
logger = logging.getLogger(__name__)

gps = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

logger.info("Reading GPS...")

while True:
    try:
        line = gps.readline().decode('utf-8', errors='ignore')

        if line.startswith("$GPRMC"):   # best for speed + location
            msg = pynmea2.parse(line)

            lat = msg.latitude
            lng = msg.longitude
            speed_knots = msg.spd_over_grnd

            # convert knots → km/h
            speed_kmh = float(speed_knots) * 1.852 if speed_knots else 0

            logger.info("Latitude :", lat)
            logger.info("Longitude:", lng)
            logger.info("Speed    :", round(speed_kmh, 2), "km/h")
            logger.info("----------------------")

    except Exception:
        pass