import serial
import pynmea2

gps = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

print("Reading GPS...")

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

            print("Latitude :", lat)
            print("Longitude:", lng)
            print("Speed    :", round(speed_kmh, 2), "km/h")
            print("----------------------")

    except Exception:
        pass