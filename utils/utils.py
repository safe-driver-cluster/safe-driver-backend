import pytz
from datetime import datetime
import os
import sys


def resource_path(relative_path):
    """Get the correct path whether running as script or compiled exe."""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller extracts files here at runtime
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def get_app_dir():
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller exe - use exe's directory
        return os.path.dirname(sys.executable)
    return get_project_dir()

def get_project_dir():
    """Return the writable application root for runtime files."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_audio_dir():
    """Return the local voice-alert audio directory."""
    return os.path.join(get_project_dir(), "audio")


def load_runtime_env():
    """Load bundled .env, then external .env beside the exe/project."""
    from dotenv import load_dotenv

    loaded_paths = []
    bundled_env = resource_path(".env")
    if os.path.exists(bundled_env):
        load_dotenv(bundled_env, override=False)
        loaded_paths.append(bundled_env)

    external_env = os.path.join(get_app_dir(), ".env")
    if os.path.exists(external_env) and os.path.abspath(external_env) != os.path.abspath(bundled_env):
        load_dotenv(external_env, override=True)
        loaded_paths.append(external_env)

    return loaded_paths


def now():
    """Return current timestamp in Sri Lanka time in ISO 8601 format"""
    # Get Sri Lanka timezone
    sri_lanka_tz = pytz.timezone('Asia/Colombo')
    return datetime.now(sri_lanka_tz).isoformat()

def print_banner(logger):
    """Print application banner from banner.txt file"""
    current_file = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(utils_dir)
    # banner_path = os.path.join(project_root, 'banner.txt')
    banner_path = resource_path('banner.txt')
    
    try:
        if os.path.exists(banner_path):
            with open(banner_path, 'r', encoding='utf-8') as f:
                banner_content = f.read()
                
                # Also write to log file (with timestamps)
                for line in banner_content.split('\n'):
                    logger.info(line)

        else:
            logger.warning(f"Banner file not found at: {banner_path}")
            
    except Exception as e:
        logger.error(f"Error printing banner: {e}", exc_info=True)

    # Print version and copyright info
    import config.config as config

    new_line1 = f"                   DRIVER MONITORING SYSTEM : Version : {config.VERSION_NO.strip()}"
    new_line2 = f"            POWERED BY CODE CRAFTERS | ALL RIGHTS RESEREVED © {datetime.now().year}"
    logger.info(new_line1)
    logger.info(new_line2+"\n")

def update_local_config_from_firestore(firestore_data: dict) -> dict:
    """
    Update local configuration values from Firestore data
    
    Args:
        firestore_data (dict): Configuration data retrieved from Firestore
        
    Returns:
        dict: Result with success status and updated configuration count
    """
    import logging
    import config.config as config
    
    logger = logging.getLogger(__name__)
    
    try:
        if not firestore_data or 'raw_configurations' not in firestore_data:
            logger.warning("No valid configuration data found in Firestore response")
            return {
                'success': False,
                'message': 'No valid configuration data found',
                'updated_count': 0
            }
        
        raw_configs = firestore_data.get('raw_configurations', {})
        updated_count = 0
        failed_updates = []
        
        # Update each configuration variable in the config module
        for config_name, config_value in raw_configs.items():
            try:
                if config_name == "GPS_SERIAL_PORT":
                    env_port = os.getenv("GPS_SERIAL_PORT")
                    simulator_port = getattr(config, "SIMULATED_GPS_SERIAL_PORT", "")
                    simulator_active = bool(
                        simulator_port and os.path.exists(simulator_port)
                    )
                    if env_port or simulator_active:
                        reason = (
                            "GPS_SERIAL_PORT environment variable"
                            if env_port
                            else f"active GPS simulator at {simulator_port}"
                        )
                        logger.info(
                            "Keeping runtime GPS_SERIAL_PORT=%s; ignoring Firestore value %s because of %s",
                            getattr(config, config_name, None),
                            config_value,
                            reason,
                        )
                        continue

                # Check if the configuration exists in the local config module
                if hasattr(config, config_name):
                    # Get the current local value for comparison
                    current_value = getattr(config, config_name)
                    
                    # Only update if values are different
                    if current_value != config_value:
                        # Set the new value in the config module
                        setattr(config, config_name, config_value)
                        updated_count += 1
                        logger.info(f"Updated {config_name}: {current_value} -> {config_value}")
                    else:
                        logger.debug(f"Config {config_name} already up to date")
                else:
                    # Configuration doesn't exist locally - could be new
                    setattr(config, config_name, config_value)
                    updated_count += 1
                    logger.info(f"Added new configuration {config_name}: {config_value}")
                    
            except Exception as e:
                failed_updates.append(f"{config_name}: {str(e)}")
                logger.error(f"Failed to update configuration {config_name}: {e}")
        
        # Log summary
        if updated_count > 0:
            logger.info(f"Successfully updated {updated_count} configuration values from Firestore")
        
        if failed_updates:
            logger.warning(f"Failed to update {len(failed_updates)} configurations: {failed_updates}")
        
        return {
            'success': True,
            'message': f'Updated {updated_count} configurations successfully',
            'updated_count': updated_count,
            'total_configs': len(raw_configs),
            'failed_updates': failed_updates,
            'firestore_version': firestore_data.get('version', 'unknown'),
            'last_updated': firestore_data.get('last_updated', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"Error updating local config from Firestore: {e}", exc_info=True)
        return {
            'success': False,
            'message': f'Failed to update local configuration: {str(e)}',
            'updated_count': 0
        }


def update_local_settings_from_firestore(firestore_data: dict) -> dict:
    """
    Update local device-specific settings from Firestore data.

    Args:
        firestore_data (dict): Settings document retrieved from settings/{device_mac}

    Returns:
        dict: Result with success status and updated settings count
    """
    import logging
    import config.settings as settings

    logger = logging.getLogger(__name__)

    try:
        if not firestore_data:
            logger.warning("No valid settings data found in Firestore response")
            return {
                'success': False,
                'message': 'No valid settings data found',
                'updated_count': 0
            }

        updated_count = 0
        failed_updates = []

        for setting_name, setting_value in firestore_data.items():
            try:
                if setting_name == "SYSTEM":
                    logger.info("Keeping runtime SYSTEM=%s; ignoring Firestore value %s", settings.SYSTEM, setting_value)
                    continue

                if not setting_name.isupper():
                    logger.debug("Skipping non-setting field from Firestore settings: %s", setting_name)
                    continue

                setting_exists = hasattr(settings, setting_name)
                current_value = getattr(settings, setting_name, None)
                if setting_exists and current_value == setting_value:
                    logger.debug("Setting %s already up to date", setting_name)
                    continue

                setattr(settings, setting_name, setting_value)
                updated_count += 1

                if setting_exists:
                    logger.info("Updated setting %s: %s -> %s", setting_name, current_value, setting_value)
                else:
                    logger.info("Added new setting %s: %s", setting_name, setting_value)

            except Exception as e:
                failed_updates.append(f"{setting_name}: {str(e)}")
                logger.error("Failed to update setting %s: %s", setting_name, e)

        if updated_count > 0:
            logger.info("Successfully updated %s settings from Firestore", updated_count)

        if failed_updates:
            logger.warning("Failed to update %s settings: %s", len(failed_updates), failed_updates)

        return {
            'success': True,
            'message': f'Updated {updated_count} settings successfully',
            'updated_count': updated_count,
            'total_settings': len(firestore_data),
            'failed_updates': failed_updates
        }

    except Exception as e:
        logger.error("Error updating local settings from Firestore: %s", e, exc_info=True)
        return {
            'success': False,
            'message': f'Failed to update local settings: {str(e)}',
            'updated_count': 0
        }
