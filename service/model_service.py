import uuid
import logging
import database.db_helper as db_helper
from beans.bean import ApiResponse, ResponseData, BehaviorResponseData

logger = logging.getLogger(__name__)

def get_mac_address() -> str:
    """
    Get the MAC address of the device.
    
    Returns:
        str: MAC address in format 'XX:XX:XX:XX:XX:XX'
    """
    try:
        # Get MAC address as integer
        mac_num = uuid.getnode()
        
        # Convert to hex string with colons
        mac_hex = ':'.join(['{:02x}'.format((mac_num >> elements) & 0xff) 
                           for elements in range(0, 8*6, 8)][::-1])
        
        logger.info(f"MAC Address retrieved: {mac_hex}")
        return mac_hex.upper()
    
    except Exception as e:
        logger.error(f"Error retrieving MAC address: {e}")
        return None


def check_device_registration() -> dict:
    """
    Check if device is registered by retrieving MAC address.
    
    Returns:
        dict: Contains MAC address and registration status
    """
    mac_address = get_mac_address_alternative()

    is_registered = db_helper.is_registered_device(mac_address) if mac_address else False
    
    if mac_address:
        return {
            "mac_address": mac_address,
            "status": "retrieved",
            "registered": is_registered
        }
    else:
        return {
            "mac_address": None,
            "status": "error",
            "registered": False
        }


# Alternative method using getmac library (more reliable)
def get_mac_address_alternative() -> str:
    """
    Get MAC address using getmac library (install: pip install getmac)
    More reliable across different platforms.
    
    Returns:
        str: MAC address in format 'XX:XX:XX:XX:XX:XX'
    """
    try:
        from getmac import get_mac_address as gma
        
        mac = gma()
        if mac:
            logger.info(f"MAC Address retrieved: {mac}")
            return mac.upper()
        else:
            logger.warning("Could not retrieve MAC address")
            return None
            
    except ImportError:
        logger.error("getmac library not installed. Install with: pip install getmac")
        return get_mac_address()  # Fallback to uuid method
    except Exception as e:
        logger.error(f"Error retrieving MAC address: {e}")
        return None


def register_device(device_info: dict) -> bool:
    """
    Register the device in the database.
    
    Args:
        device_info (dict): Information about the device to register
        
    Returns:
        bool: True if registration successful, False otherwise
    """
    mac_address = device_info.get("mac_address")
    
    result = db_helper.register_device(mac=mac_address)
    
    if result.get('success'):
        logger.info(f"Device with MAC {mac_address} registered successfully")
        return True
    else:
        logger.error(f"Failed to register device with MAC {mac_address}: {result.get('message')}")
        return False
    
def update_device_status(mac: str, status: str):
    """
    Update the status of a registered device.
    
    Args:
        mac (str): MAC address of the device
        status (str): New status to set
        
    Returns:
        bool: True if update successful, False otherwise
    """
    result = db_helper.update_device_status(mac, status)
    
    if result.get('success'):
        logger.info(f"Device {mac} status updated to {status}")
    else:
        logger.error(f"Failed to update status for device {mac}: {result.get('message')}")
