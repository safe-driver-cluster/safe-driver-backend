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

def perform_voice_alerts(message):
    """Perform voice alerts using system TTS"""
    try:
        # For Windows
        if os.name == 'nt':
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        # For macOS
        elif os.uname().sysname == 'Darwin':
            os.system(f'say "{message}"')
        # For Linux
        else:
            os.system(f'espeak "{message}"')
    except Exception as e:
        print(f"Error performing voice alert: {e}")
