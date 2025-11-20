import logging
from firebase_admin import firestore
from datetime import datetime
from typing import Dict, List, Optional, Any
import utils.utils as utils
import config.config as config
import cv2
import inspect

logger = logging.getLogger(__name__)

# Initialize Firestore client - will be set after Firebase is initialized
db = None

class FirestoreHelper:
    """Helper class for Firebase Firestore operations"""
    
    def __init__(self):
        self.db = None
    
    def _ensure_db_initialized(self):
        """Ensure Firestore client is initialized"""
        if self.db is None:
            try:
                self.db = firestore.client()
            except Exception as e:
                logger.error(f"Failed to initialize Firestore client: {e}")
                raise
        
    def save_model_configurations_to_firestore(self) -> Dict:
        """
        Save all configurations from config.py to Firestore under 'model_configurations' document in 'configs' collection.
        
        Returns:
            dict: Operation result with success status and message
        """
        try:
            self._ensure_db_initialized()
            # Get all configuration variables from config module
            config_vars = {}
            
            # Get all attributes from config module
            for name in dir(config):
                # Skip private attributes and functions
                if not name.startswith('_') and not inspect.isfunction(getattr(config, name)) and not inspect.ismodule(getattr(config, name)):
                    value = getattr(config, name)
                    
                    # Convert non-serializable values to serializable format
                    if isinstance(value, tuple):
                        # Convert tuples to lists for JSON serialization
                        config_vars[name] = list(value)
                    elif hasattr(cv2, name.split('_')[0]) and str(type(value)).startswith("<class 'int'>"):
                        # Handle OpenCV constants (they're integers)
                        config_vars[name] = value
                    elif isinstance(value, (str, int, float, bool, list, dict)):
                        # Directly serializable types
                        config_vars[name] = value
                    else:
                        # Convert other types to string representation
                        config_vars[name] = str(value)
            
            # Organize configurations by category based on comments in config.py
            organized_configs = {
                'system_configuration': {
                    'VERSION_NO': config_vars.get('VERSION_NO')
                },
                'drowsiness_detection_thresholds': {
                    'EYE_CLOSED_THRESH': config_vars.get('EYE_CLOSED_THRESH'),
                    'EYE_PARTIAL_THRESH': config_vars.get('EYE_PARTIAL_THRESH'),
                    'MICROSLEEP_SEC': config_vars.get('MICROSLEEP_SEC'),
                    'PERCLOS_WIN_SEC': config_vars.get('PERCLOS_WIN_SEC'),
                    'PERCLOS_DROWSY': config_vars.get('PERCLOS_DROWSY'),
                    'YAWN_THRESH': config_vars.get('YAWN_THRESH'),
                    'YAWN_MIN_SEC': config_vars.get('YAWN_MIN_SEC'),
                    'EYE_CLOSURE_FREQ_WIN': config_vars.get('EYE_CLOSURE_FREQ_WIN'),
                    'EYE_CLOSURE_FREQ_THRESH': config_vars.get('EYE_CLOSURE_FREQ_THRESH'),
                    'MIN_CLOSURE_DURATION': config_vars.get('MIN_CLOSURE_DURATION'),
                    'BLINK_MAX_DURATION': config_vars.get('BLINK_MAX_DURATION'),
                    'CLOSURE_DEBOUNCE_TIME': config_vars.get('CLOSURE_DEBOUNCE_TIME')
                },
                'head_pose_detection_thresholds': {
                    'HEAD_YAW_THRESH_LEFT': config_vars.get('HEAD_YAW_THRESH_LEFT'),
                    'HEAD_YAW_THRESH_RIGHT': config_vars.get('HEAD_YAW_THRESH_RIGHT'),
                    'HEAD_TURN_DISTRACTION_SEC': config_vars.get('HEAD_TURN_DISTRACTION_SEC'),
                    'SHOW_HEAD_POSE_DETAILS': config_vars.get('SHOW_HEAD_POSE_DETAILS')
                },
                'ui_layout_parameters': {
                    'WINDOW_NAME': config_vars.get('WINDOW_NAME'),
                    'ROW_SIZE': config_vars.get('ROW_SIZE'),
                    'LEFT_MARGIN': config_vars.get('LEFT_MARGIN'),
                    'LABEL_PADDING_WIDTH': config_vars.get('LABEL_PADDING_WIDTH'),
                    'FPS_AVG_FRAME_COUNT': config_vars.get('FPS_AVG_FRAME_COUNT'),
                    'SCROLL_STEP': config_vars.get('SCROLL_STEP')
                },
                'display_control_flags': {
                    'SHOW_BLENDSHAPES': config_vars.get('SHOW_BLENDSHAPES'),
                    'SHOW_FACE_MESH': config_vars.get('SHOW_FACE_MESH'),
                    'SHOW_FPS': config_vars.get('SHOW_FPS'),
                    'SHOW_METRICS': config_vars.get('SHOW_METRICS'),
                    'SHOW_WARNINGS': config_vars.get('SHOW_WARNINGS'),
                    'ENABLE_VOICE_ALERTS': config_vars.get('ENABLE_VOICE_ALERTS')
                },
                'display_settings': {
                    'fps_display': {
                        'FPS_FONT': config_vars.get('FPS_FONT'),
                        'FPS_FONT_SIZE': config_vars.get('FPS_FONT_SIZE'),
                        'FPS_FONT_THICKNESS': config_vars.get('FPS_FONT_THICKNESS'),
                        'FPS_COLOR': config_vars.get('FPS_COLOR'),
                        'FPS_TEXT_FORMAT': config_vars.get('FPS_TEXT_FORMAT'),
                        'FPS_Y_OFFSET': config_vars.get('FPS_Y_OFFSET')
                    },
                    'metrics_box': {
                        'METRICS_PADDING': config_vars.get('METRICS_PADDING'),
                        'METRICS_WIDTH': config_vars.get('METRICS_WIDTH'),
                        'METRICS_HEIGHT': config_vars.get('METRICS_HEIGHT'),
                        'METRICS_Y_OFFSET': config_vars.get('METRICS_Y_OFFSET'),
                        'METRICS_CORNER_RADIUS': config_vars.get('METRICS_CORNER_RADIUS'),
                        'METRICS_BG_COLOR': config_vars.get('METRICS_BG_COLOR'),
                        'METRICS_BG_OPACITY': config_vars.get('METRICS_BG_OPACITY'),
                        'METRICS_FONT': config_vars.get('METRICS_FONT'),
                        'METRICS_FONT_SIZE': config_vars.get('METRICS_FONT_SIZE'),
                        'METRICS_FONT_THICKNESS': config_vars.get('METRICS_FONT_THICKNESS'),
                        'METRICS_TEXT_COLOR': config_vars.get('METRICS_TEXT_COLOR')
                    },
                    'warning_display': {
                        'WARNING_FONT': config_vars.get('WARNING_FONT'),
                        'WARNING_FONT_SIZE': config_vars.get('WARNING_FONT_SIZE'),
                        'WARNING_FONT_THICKNESS': config_vars.get('WARNING_FONT_THICKNESS'),
                        'WARNING_COLOR': config_vars.get('WARNING_COLOR'),
                        'WARNING_Y_POSITION': config_vars.get('WARNING_Y_POSITION'),
                        'WARNING_RIGHT_MARGIN': config_vars.get('WARNING_RIGHT_MARGIN')
                    },
                    'blendshapes_display': {
                        'BLENDSHAPE_FONT': config_vars.get('BLENDSHAPE_FONT'),
                        'BLENDSHAPE_FONT_SIZE': config_vars.get('BLENDSHAPE_FONT_SIZE'),
                        'BLENDSHAPE_FONT_THICKNESS': config_vars.get('BLENDSHAPE_FONT_THICKNESS'),
                        'BLENDSHAPE_TEXT_COLOR': config_vars.get('BLENDSHAPE_TEXT_COLOR'),
                        'BLENDSHAPE_BAR_COLOR': config_vars.get('BLENDSHAPE_BAR_COLOR'),
                        'BLENDSHAPE_BAR_HEIGHT': config_vars.get('BLENDSHAPE_BAR_HEIGHT'),
                        'BLENDSHAPE_GAP_BETWEEN_BARS': config_vars.get('BLENDSHAPE_GAP_BETWEEN_BARS'),
                        'BLENDSHAPE_TEXT_GAP': config_vars.get('BLENDSHAPE_TEXT_GAP'),
                        'BLENDSHAPE_X_OFFSET': config_vars.get('BLENDSHAPE_X_OFFSET'),
                        'BLENDSHAPE_Y_START': config_vars.get('BLENDSHAPE_Y_START'),
                        'BLENDSHAPE_TEXT_FORMAT': config_vars.get('BLENDSHAPE_TEXT_FORMAT')
                    },
                    'head_pose_display': {
                        'HEAD_POSE_DETAILS_Y_OFFSET': config_vars.get('HEAD_POSE_DETAILS_Y_OFFSET'),
                        'HEAD_POSE_FONT_SIZE': config_vars.get('HEAD_POSE_FONT_SIZE'),
                        'HEAD_POSE_COLOR': config_vars.get('HEAD_POSE_COLOR')
                    }
                },
                'text_labels_and_messages': {
                    'metrics_labels': {
                        'LABEL_PERCLOS': config_vars.get('LABEL_PERCLOS'),
                        'LABEL_BLINKS': config_vars.get('LABEL_BLINKS'),
                        'LABEL_CLOSURES': config_vars.get('LABEL_CLOSURES'),
                        'LABEL_YAWNS': config_vars.get('LABEL_YAWNS'),
                        'LABEL_MICROSLEEPS': config_vars.get('LABEL_MICROSLEEPS'),
                        'LABEL_DROWSY_EVENTS': config_vars.get('LABEL_DROWSY_EVENTS'),
                        'LABEL_HEAD_POSE': config_vars.get('LABEL_HEAD_POSE')
                    },
                    'metrics_positions': {
                        'PERCLOS_Y_OFFSET': config_vars.get('PERCLOS_Y_OFFSET'),
                        'BLINKS_Y_OFFSET': config_vars.get('BLINKS_Y_OFFSET'),
                        'CLOSURES_Y_OFFSET': config_vars.get('CLOSURES_Y_OFFSET'),
                        'YAWNS_Y_OFFSET': config_vars.get('YAWNS_Y_OFFSET'),
                        'MICROSLEEPS_Y_OFFSET': config_vars.get('MICROSLEEPS_Y_OFFSET'),
                        'DROWSY_EVENTS_Y_OFFSET': config_vars.get('DROWSY_EVENTS_Y_OFFSET'),
                        'HEAD_POSE_Y_OFFSET': config_vars.get('HEAD_POSE_Y_OFFSET')
                    },
                    'warning_messages': {
                        'WARNING_MICROSLEEP': config_vars.get('WARNING_MICROSLEEP'),
                        'WARNING_YAWNING': config_vars.get('WARNING_YAWNING'),
                        'WARNING_FREQUENT_CLOSURES': config_vars.get('WARNING_FREQUENT_CLOSURES'),
                        'WARNING_DROWSY': config_vars.get('WARNING_DROWSY'),
                        'WARNING_PERCLOS': config_vars.get('WARNING_PERCLOS'),
                        'WARNING_DISTRACTION': config_vars.get('WARNING_DISTRACTION'),
                        'WARNING_MOBILE_USE': config_vars.get('WARNING_MOBILE_USE'),
                        'WARNING_SMOKING': config_vars.get('WARNING_SMOKING'),
                        'WARNING_HEAD_TURN': config_vars.get('WARNING_HEAD_TURN')
                    },
                    'console_messages': {
                        'CONSOLE_MICROSLEEP': config_vars.get('CONSOLE_MICROSLEEP'),
                        'CONSOLE_YAWN': config_vars.get('CONSOLE_YAWN'),
                        'CONSOLE_FREQUENT_CLOSURES': config_vars.get('CONSOLE_FREQUENT_CLOSURES'),
                        'CONSOLE_DROWSY': config_vars.get('CONSOLE_DROWSY'),
                        'CONSOLE_PERCLOS_REACHED': config_vars.get('CONSOLE_PERCLOS_REACHED'),
                        'CONSOLE_DISTRACTION': config_vars.get('CONSOLE_DISTRACTION'),
                        'CONSOLE_MOBILE_USE': config_vars.get('CONSOLE_MOBILE_USE'),
                        'CONSOLE_SMOKING': config_vars.get('CONSOLE_SMOKING'),
                        'CONSOLE_HEAD_TURN': config_vars.get('CONSOLE_HEAD_TURN'),
                        'CONSOLE_FACE_LOSS': config_vars.get('CONSOLE_FACE_LOSS')
                    },
                    'voice_alert_messages': {
                        'VOICE_ALERT_MICROSLEEP': config_vars.get('VOICE_ALERT_MICROSLEEP'),
                        'VOICE_ALERT_YAWNING': config_vars.get('VOICE_ALERT_YAWNING'),
                        'VOICE_ALERT_DROWSY': config_vars.get('VOICE_ALERT_DROWSY'),
                        'VOICE_ALERT_DISTRACTION': config_vars.get('VOICE_ALERT_DISTRACTION'),
                        'VOICE_ALERT_HEAD_TURN': config_vars.get('VOICE_ALERT_HEAD_TURN'),
                        'VOICE_ALERT_PERCLOS': config_vars.get('VOICE_ALERT_PERCLOS')
                    }
                },
                'behavior_data_types': {
                    'BEHAVIOR_FREQUENT_CLOSURES': config_vars.get('BEHAVIOR_FREQUENT_CLOSURES'),
                    'BEHAVIOR_MICROSLEEP': config_vars.get('BEHAVIOR_MICROSLEEP'),
                    'BEHAVIOR_YAWN': config_vars.get('BEHAVIOR_YAWN'),
                    'BEHAVIOR_DROWSY': config_vars.get('BEHAVIOR_DROWSY'),
                    'BEHAVIOR_PERCLOS_REACHED': config_vars.get('BEHAVIOR_PERCLOS_REACHED'),
                    'BEHAVIOR_DISTRACTION': config_vars.get('BEHAVIOR_DISTRACTION'),
                    'BEHAVIOR_MOBILE_USE': config_vars.get('BEHAVIOR_MOBILE_USE'),
                    'BEHAVIOR_SMOKING': config_vars.get('BEHAVIOR_SMOKING'),
                    'BEHAVIOR_HEAD_TURN': config_vars.get('BEHAVIOR_HEAD_TURN')
                },
                'other_settings': {
                    'LABEL_BG_COLOR': config_vars.get('LABEL_BG_COLOR'),
                    'CAMERA_ERROR_MSG': config_vars.get('CAMERA_ERROR_MSG')
                }
            }
            
            # Prepare the document data with metadata
            document_data = {
                'configurations': organized_configs,
                'raw_configurations': config_vars,  # Also save raw config for completeness
                'last_updated': firestore.SERVER_TIMESTAMP,
                'created_at': firestore.SERVER_TIMESTAMP,
                'version': config_vars.get('VERSION_NO', '1.0.0'),
                'total_config_count': len(config_vars)
            }
            
            # Save to Firestore: configs collection -> model_configurations document
            doc_ref = self.db.collection('configs').document('model_configurations')
            doc_ref.set(document_data, merge=True)
            
            logger.info(f"Successfully saved {len(config_vars)} configurations to Firestore")
            return {
                'success': True,
                'message': f'Successfully saved {len(config_vars)} configurations to Firestore',
                'config_count': len(config_vars),
                'categories': list(organized_configs.keys()),
                'document_path': 'configs/model_configurations'
            }
            
        except Exception as e:
            logger.error(f"Error saving model configurations to Firestore: {e}", exc_info=True)
            return {
                'success': False,
                'message': f'Failed to save configurations: {str(e)}',
                'config_count': 0
            }
    
    def get_model_configurations_from_firestore(self) -> Dict:
        """
        Retrieve model configurations from Firestore.
        
        Returns:
            dict: Configuration data or empty dict if not found
        """
        try:
            self._ensure_db_initialized()
            doc_ref = self.db.collection('configs').document('model_configurations')
            doc = doc_ref.get()
            
            if doc.exists:
                config_data = doc.to_dict()
                logger.info("Successfully retrieved model configurations from Firestore")
                return config_data
            else:
                logger.warning("No model configurations found in Firestore")
                return {}
                
        except Exception as e:
            logger.error(f"Error retrieving model configurations from Firestore: {e}")
            return {}
    
    def update_specific_configuration_firestore(self, config_category: str, config_name: str, config_value: Any) -> Dict:
        """
        Update a specific configuration value in Firestore.
        
        Args:
            config_category (str): Category of the configuration (e.g., 'drowsiness_detection_thresholds')
            config_name (str): Name of the configuration variable
            config_value (Any): New value for the configuration
            
        Returns:
            dict: Update result
        """
        try:
            self._ensure_db_initialized()
            doc_ref = self.db.collection('configs').document('model_configurations')
            
            # Update both organized and raw configurations
            update_data = {
                f'configurations.{config_category}.{config_name}': config_value,
                f'raw_configurations.{config_name}': config_value,
                'last_updated': firestore.SERVER_TIMESTAMP
            }
            
            doc_ref.update(update_data)
            
            logger.info(f"Successfully updated configuration {config_name} in category {config_category}")
            return {
                'success': True,
                'message': f'Successfully updated {config_name}',
                'config_name': config_name,
                'config_value': config_value,
                'category': config_category
            }
            
        except Exception as e:
            logger.error(f"Error updating configuration {config_name}: {e}")
            return {
                'success': False,
                'message': f'Failed to update {config_name}: {str(e)}'
            }

# Create a singleton instance
firestore_helper = FirestoreHelper()