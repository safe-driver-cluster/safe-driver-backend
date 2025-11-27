import logging
from firebase_admin import firestore
from datetime import datetime
from typing import Dict, List, Optional, Any
from database import db_helper
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
                        
            # Prepare the document data with metadata
            document_data = {
                # 'configurations': organized_configs,
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
                # f'configurations.{config_category}.{config_name}': config_value,
                f'raw_configurations.{config_name}': config_value,
                'last_updated': firestore.SERVER_TIMESTAMP
            }
            
            doc_ref.update(update_data)
            
            # logger.info(f"Successfully updated configuration {config_name} in category {config_category}")
            logger.info(f"Successfully updated configuration : {config_name} --> {config_value}")
            return {
                'success': True,
                'message': f'Successfully updated {config_name}',
                'config_name': config_name,
                'config_value': config_value,
                # 'category': config_category
            }
            
        except Exception as e:
            logger.error(f"Error updating configuration {config_name}: {e}")
            return {
                'success': False,
                'message': f'Failed to update {config_name}: {str(e)}'
            }
        

    def save_alert_history_to_firestore(self, mac: str, behavior_data: dict) -> Dict:
        """
        Save an alert history record to Firestore under 'alert_histories' collection.
        
        Args:
            alert_data (dict): Alert data to be saved
            
        Returns:
            dict: Operation result with success status and message
        """
        try:

            number_plate = db_helper.get_vehicle_reg_no(mac)

            # Save to Firestore: alerts collection / {mac} document / date collection (2025-11-27) / auto-generated document
            self._ensure_db_initialized()
            date_str = utils.now().split('T')[0]

            doc_ref = self.db.collection('alerts').document(mac).collection(date_str).document()
            alert_data = {
                'tag': behavior_data.get('tag'),
                'type': behavior_data.get('type'),
                'message': behavior_data.get('message'),
                'time': behavior_data.get('time', utils.now()),
                'number_plate':number_plate,
            }
            doc_ref.set(alert_data)
            
        except Exception as e:
            logger.error(f"Error saving alert history to Firestore: {e}")
            return {
                'success': False,
                'message': f'Failed to save alert history: {str(e)}'
            }

    def register_driver_fingerprint(self, driver_id: str, fingerprint_data: any) -> Dict:
        """
        Register a driver's fingerprint data in Firestore.
        
        Args:
            driver_id (str): Unique identifier for the driver
            """
        try:
            # firestore/drivers/document{driver_id}/fingerprint_id
            self._ensure_db_initialized()
            doc_ref = self.db.collection('drivers').document(driver_id)
            doc_ref.set({
                'fingerprint_id': fingerprint_data,
                'fp_registered_at': firestore.SERVER_TIMESTAMP
            }, merge=True)

            
        except Exception as e:
            logger.error(f"Error registering fingerprint for driver {driver_id}: {e}")
            return {
                'success': False,
                'message': f'Failed to register fingerprint: {str(e)}'
            }
        
    # get driver_id by fingerprint
    def get_driver_by_fingerprint(self, fingerprint_data: any) -> Optional[str]:
        """
        Retrieve driver ID by fingerprint data from Firestore.
        
        Args:
            fingerprint_data (any): Fingerprint data to search for
            """
        try:
            self._ensure_db_initialized()
            drivers_ref = self.db.collection('drivers')
            query = drivers_ref.where('fingerprint_id', '==', fingerprint_data)

            # check if query has multiple results


            results = query.stream()
            
            for doc in results:
                logger.info(f"Driver found for given fingerprint: {doc.id}")
                return doc.id  # Return the driver_id
            
            logger.info("No driver found for the given fingerprint")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving driver by fingerprint: {e}")
            return None

# Create a singleton instance
firestore_helper = FirestoreHelper()