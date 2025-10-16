import pytz
from datetime import datetime

def now():
    """Return current timestamp in Sri Lanka time in ISO 8601 format"""
    # Get Sri Lanka timezone
    sri_lanka_tz = pytz.timezone('Asia/Colombo')
    return datetime.now(sri_lanka_tz).isoformat()