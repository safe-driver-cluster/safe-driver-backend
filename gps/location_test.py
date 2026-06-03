# import requests

# response = requests.get("https://ipinfo.io/json")
# data = response.json()

# loc = data['loc'].split(',')
# lat = float(loc[0])
# lng = float(loc[1])

# print(f"Latitude : {lat}")
# print(f"Longitude: {lng}")
# print(f"City     : {data.get('city')}")
# print(f"Region   : {data.get('region')}")

from math import radians, sin, cos, sqrt, atan2
import requests, time

def get_ip_location():
    r = requests.get("https://ipinfo.io/json")
    lat, lng = r.json()['loc'].split(',')
    return float(lat), float(lng)

def haversine(lat1, lng1, lat2, lng2):
    R = 6371000  # Earth radius in meters
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

lat1, lng1 = get_ip_location()
t1 = time.time()

time.sleep(5)

lat2, lng2 = get_ip_location()
t2 = time.time()

distance_m = haversine(lat1, lng1, lat2, lng2)
speed_kmh = (distance_m / (t2 - t1)) * 3.6

print(f"Latitude : {lat2}")
print(f"Longitude: {lng2}")
print(f"Speed: {round(speed_kmh, 2)} km/h")