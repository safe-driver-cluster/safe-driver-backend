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

                new_line1 = f"                   DRIVER MONITORING SYSTEM : Version : {config.VERSION_NO.strip()}"
                new_line2 = f"            POWERED BY CODE CRAFTERS | ALL RIGHTS RESEREVED © {datetime.now().year}"

                logger.info(new_line1)
                logger.info(new_line2+"\n")

        else:
            logger.warning(f"Banner file not found at: {banner_path}")
            # Fallback banner if file doesn't exist
            logger.info("=" * 80)
            logger.info(f"SAFE DRIVER MONITORING SYSTEM : {config.VERSION_NO}")
            logger.info("Real-time Driver Drowsiness Detection")
            logger.info("=" * 80)
            
    except Exception as e:
        logger.error(f"Error printing banner: {e}", exc_info=True)
        logger.info("=" * 80)
        logger.info(f"SAFE DRIVER MONITORING SYSTEM : {config.VERSION_NO}")
        logger.info("=" * 80)