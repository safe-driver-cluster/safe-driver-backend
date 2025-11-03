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
                
                # Print to console without timestamp
                # print(banner_content)
                # print()  # Empty line for spacing
                
                # Also write to log file (with timestamps)
                for line in banner_content.split('\n'):
                    if line.strip():
                        # Check if line contains version placeholder
                        if ":" in line:
                            line_parts = line.split(":", 1)
                            # Format with version number
                            new_line = f"{line_parts[0].strip()} : Version : {config.VERSION_NO.strip()}"
                            logger.info(new_line)
                        elif "©" in line:
                            line_parts = line.split("©", 1)
                            new_line = f"{line_parts[0].strip()}© {datetime.now().year}"
                            logger.info(new_line)
                        else:
                            # Log line as-is
                            logger.info(line)
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