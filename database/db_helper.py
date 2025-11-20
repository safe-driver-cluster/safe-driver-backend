import logging
from firebase_admin import db
import utils.utils as utils
from beans.bean import ApiResponse, ResponseData, BehaviorResponseData
import config.config as config

logger = logging.getLogger(__name__)

def is_registered_device(mac: str) -> bool:
    """
    Check if a device with the given MAC address is registered in Firebase Realtime Database.
    
    Args:
        mac (str): MAC address to check (used as key)
        
    Returns:
        bool: True if device is registered, False otherwise
    """
    try:
        # Reference to the specific device using MAC as key
        ref = db.reference(f'devices/{mac}')
        
        # Get device data
        device_data = ref.get()
        
        if device_data is None:
            logger.info(f"Device with MAC {mac} is not registered")
            return False
        
        # Check if 'registered' field is True
        is_registered = device_data.get('is_registered', False)
        
        return is_registered
        
    except Exception as e:
        logger.error(f"Error checking device registration for MAC {mac}: {e}")
        return False


def get_device_by_mac(mac: str) -> dict:
    """
    Get device details by MAC address.
    
    Args:
        mac (str): MAC address to search for
        
    Returns:
        dict: Device data if found, None otherwise
    """
    try:
        ref = db.reference(f'devices/{mac}')
        device_data = ref.get()
        
        if device_data is None:
            logger.warning(f"No device found with MAC {mac}")
            return None
        
        logger.info(f"Found device with MAC {mac}")
        return {
            'mac': mac,
            **device_data
        }
        
    except Exception as e:
        logger.error(f"Error retrieving device by MAC {mac}: {e}")
        return None


def register_device(mac: str, reg_no: str = None) -> dict:
    """
    Register a new device in Firebase Realtime Database.
    
    Args:
        mac (str): MAC address of the device (used as key)
        reg_no (str): Optional registration number
        
    Returns:
        dict: Registration result
    """
    try:
        ref = db.reference(f'devices/{mac}')
        
        # Check if device already exists
        existing_device = ref.get()
        
        if existing_device and existing_device.get('is_registered'):
            logger.warning(f"Device with MAC {mac} is already registered")
            return {
                'success': False,
                'message': 'Device already registered',
                'device': existing_device
            }
        
        # Create/update device entry
        current_time = utils.now()
        device_data = {
            'vehicle_reg_no': reg_no or '',
            'is_registered': True,
            'is_verified': False,
            'registered_date_time': current_time,
            'verified_date_time': None,
            'last_updated_date_time': current_time,
            'last_active_date_time': current_time,
            'status': 'active',
        }
        
        ref.set(device_data)
        
        logger.info(f"Device registered successfully: {mac}")
        return {
            'success': True,
            'mac': mac,
            'message': 'Device registered successfully',
            'device': device_data
        }
        
    except Exception as e:
        logger.error(f"Error registering device with MAC {mac}: {e}")
        return {
            'success': False,
            'message': str(e)
        }


def update_device_status(mac: str, status: str) -> dict:
    """
    Update device status.
    
    Args:
        mac (str): MAC address of the device
        status (str): New status (e.g., 'active', 'inactive', 'suspended')
        
    Returns:
        dict: Update result
    """
    try:
        ref = db.reference(f'devices/{mac}')
        
        # Check if device exists
        device_data = ref.get()
        
        if device_data is None:
            logger.warning(f"Device with MAC {mac} not found")
            return {
                'success': False,
                'message': 'Device not found'
            }
        
        # Update status and last updated time
        ref.update({
            'status': status,
            'last_updated_date_time': utils.now()
        })
        
        return {
            'success': True,
            'mac': mac,
            'status': status,
            'message': 'Status updated successfully'
        }
        
    except Exception as e:
        logger.error(f"Error updating device status for MAC {mac}: {e}")
        return {
            'success': False,
            'message': str(e)
        }


def update_last_active(mac: str) -> dict:
    """
    Update the last active timestamp for a device.
    
    Args:
        mac (str): MAC address of the device
        
    Returns:
        dict: Update result
    """
    try:
        ref = db.reference(f'devices/{mac}')
        
        # Check if device exists
        device_data = ref.get()
        
        if device_data is None:
            logger.warning(f"Device with MAC {mac} not found")
            return {
                'success': False,
                'message': 'Device not found'
            }
        
        # Update last active time
        ref.update({
            'last_active_date_time': utils.now()
        })
        
        logger.debug(f"Device {mac} last active time updated")
        return {
            'success': True,
            'mac': mac,
            'message': 'Last active time updated'
        }
        
    except Exception as e:
        logger.error(f"Error updating last active time for MAC {mac}: {e}")
        return {
            'success': False,
            'message': str(e)
        }


def get_all_devices() -> dict:
    """
    Get all registered devices.
    
    Returns:
        dict: All devices data
    """
    try:
        ref = db.reference('devices')
        devices = ref.get()
        
        if not devices:
            logger.warning("No devices found in database")
            return {}
        
        logger.info(f"Retrieved {len(devices)} devices")
        return devices
        
    except Exception as e:
        logger.error(f"Error retrieving all devices: {e}")
        return {}


def delete_device(mac: str) -> dict:
    """
    Delete a device from the database.
    
    Args:
        mac (str): MAC address of the device
        
    Returns:
        dict: Deletion result
    """
    try:
        ref = db.reference(f'devices/{mac}')
        
        # Check if device exists
        device_data = ref.get()
        
        if device_data is None:
            logger.warning(f"Device with MAC {mac} not found")
            return {
                'success': False,
                'message': 'Device not found'
            }
        
        # Delete device
        ref.delete()
        
        logger.info(f"Device {mac} deleted successfully")
        return {
            'success': True,
            'mac': mac,
            'message': 'Device deleted successfully'
        }
        
    except Exception as e:
        logger.error(f"Error deleting device with MAC {mac}: {e}")
        return {
            'success': False,
            'message': str(e)
        }

def save_behavior_to_firebase(mac: str, behavior_data: dict):
    """
    Save behavior data to Firebase Realtime Database
    
    Args:
        mac (str): MAC address of the device
        behavior_data (dict): Behavior data containing tag, type, message, time, and data
    """
    try:
        ref = db.reference(f'alerts/{mac}')
        
        # Extract the data to save
        alert_data = {
            'tag': behavior_data.get('tag'),
            'type': behavior_data.get('type'),
            'message': behavior_data.get('message'),
            'time': behavior_data.get('time', utils.now()),
            'number_plate':"NB-9999",
        }
        
        # Save to latest
        ref.child('latest').set(alert_data)
        
        # If significant event, save to history
        ref.child('history').push(alert_data)
        
    except Exception as e:
        logger.error(f"Failed to save behavior data to Firebase for MAC {mac}: {e}", exc_info=True)
