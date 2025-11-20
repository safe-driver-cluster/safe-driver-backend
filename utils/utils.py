import pytz
from datetime import datetime
import os
import config.config as config

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
    banner_path = os.path.join(project_root, 'banner.txt')
    
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
