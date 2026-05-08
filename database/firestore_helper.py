import logging
import hashlib
from firebase_admin import firestore
from datetime import datetime
from typing import Dict, List, Optional, Any
from database import db_helper
import utils.utils as utils
import config.config as config
import cv2
import inspect
from google.cloud.firestore_v1.base_query import FieldFilter

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

    @staticmethod
    def build_fingerprint_template_id(scanner_id: str, template_position: int) -> str:
        """Build a deterministic 32-char ID for a scanner template slot."""
        normalized = f"{scanner_id.strip().lower()}:{int(template_position)}"
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
        
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

    def register_driver_fingerprint(
        self,
        driver_id: str,
        fingerprint_data: Any = None,
        scanner_id: Optional[str] = None,
        template_position: Optional[int] = None,
    ) -> Dict:
        """
        Register a driver's fingerprint data in Firestore.
        
        Args:
            driver_id (str): Unique identifier for the driver
            fingerprint_data (Any): Legacy single fingerprint identifier (optional)
            scanner_id (str): Unique scanner/device identifier (optional)
            template_position (int): Template position inside scanner DB (optional)
        """
        try:
            self._ensure_db_initialized()

            payload = {
                'fp_registered_at': firestore.SERVER_TIMESTAMP
            }

            if fingerprint_data is not None:
                payload['fingerprint_id'] = fingerprint_data

            template_id = None
            if scanner_id is not None and template_position is not None:
                template_position = int(template_position)
                template_id = self.build_fingerprint_template_id(scanner_id, template_position)
                payload.update({
                    'fingerprint_template_id': template_id,
                    'fingerprint_scanner_id': scanner_id,
                    'fingerprint_template_position': template_position,
                })

            doc_ref = self.db.collection('drivers').document(driver_id)
            doc_ref.set(payload, merge=True)

            if template_id is not None:
                # Fast reverse lookup: template -> driver.
                template_ref = self.db.collection('fingerprint_templates').document(template_id)
                template_ref.set({
                    'driver_id': driver_id,
                    'scanner_id': scanner_id,
                    'template_position': template_position,
                    'updated_at': firestore.SERVER_TIMESTAMP,
                }, merge=True)

            return {
                'success': True,
                'driver_id': driver_id,
                'fingerprint_id': fingerprint_data,
                'template_id': template_id,
                'scanner_id': scanner_id,
                'template_position': template_position,
            }

            
        except Exception as e:
            logger.error(f"Error registering fingerprint for driver {driver_id}: {e}")
            return {
                'success': False,
                'message': f'Failed to register fingerprint: {str(e)}'
            }
        
    # get driver_id by fingerprint
    def get_driver_by_fingerprint(
        self,
        fingerprint_data: Any = None,
        scanner_id: Optional[str] = None,
        template_position: Optional[int] = None,
        template_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieve driver ID by fingerprint data from Firestore.
        
        Args:
            fingerprint_data (Any): Legacy fingerprint identifier (optional)
            scanner_id (str): Scanner/device identifier (optional)
            template_position (int): Template position in scanner (optional)
            template_id (str): Precomputed template identifier (optional)
        """
        try:
            self._ensure_db_initialized()

            if template_id is None and scanner_id is not None and template_position is not None:
                template_id = self.build_fingerprint_template_id(scanner_id, int(template_position))

            if template_id is not None:
                template_doc = self.db.collection('fingerprint_templates').document(template_id).get()
                if template_doc.exists:
                    data = template_doc.to_dict() or {}
                    driver_id = data.get('driver_id')
                    if driver_id:
                        logger.info(f"Driver found for template ID: {driver_id}")
                        return driver_id

            drivers_ref = self.db.collection('drivers')
            query = drivers_ref.where(
                filter=FieldFilter('fingerprint_id', '==', fingerprint_data)
            )

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

    def delete_driver_fingerprint(
        self,
        driver_id: Optional[str] = None,
        scanner_id: Optional[str] = None,
        template_position: Optional[int] = None,
        template_id: Optional[str] = None,
        delete_legacy_fingerprint_id: bool = True,
    ) -> Dict:
        """
        Delete fingerprint mapping from Firestore.

        Supports deleting by driver_id and/or scanner template mapping.
        """
        try:
            self._ensure_db_initialized()

            if driver_id is None and template_id is None and (scanner_id is None or template_position is None):
                return {
                    'success': False,
                    'message': 'Provide driver_id or template_id or scanner_id+template_position'
                }

            resolved_template_id = template_id
            if resolved_template_id is None and scanner_id is not None and template_position is not None:
                resolved_template_id = self.build_fingerprint_template_id(scanner_id, int(template_position))

            driver_doc_data = None
            if driver_id is not None:
                driver_ref = self.db.collection('drivers').document(driver_id)
                driver_doc = driver_ref.get()
                if not driver_doc.exists:
                    return {
                        'success': False,
                        'message': f'Driver not found: {driver_id}'
                    }

                driver_doc_data = driver_doc.to_dict() or {}
                if resolved_template_id is None:
                    resolved_template_id = driver_doc_data.get('fingerprint_template_id')

                delete_payload = {
                    'fingerprint_template_id': firestore.DELETE_FIELD,
                    'fingerprint_scanner_id': firestore.DELETE_FIELD,
                    'fingerprint_template_position': firestore.DELETE_FIELD,
                    'fp_registered_at': firestore.DELETE_FIELD,
                    'fp_deleted_at': firestore.SERVER_TIMESTAMP,
                }
                if delete_legacy_fingerprint_id:
                    delete_payload['fingerprint_id'] = firestore.DELETE_FIELD

                driver_ref.update(delete_payload)

            if resolved_template_id is not None:
                template_ref = self.db.collection('fingerprint_templates').document(resolved_template_id)
                template_ref.delete()

            return {
                'success': True,
                'driver_id': driver_id,
                'template_id': resolved_template_id,
                'message': 'Fingerprint mapping deleted successfully'
            }

        except Exception as e:
            logger.error(f"Error deleting fingerprint mapping: {e}")
            return {
                'success': False,
                'message': f'Failed to delete fingerprint mapping: {str(e)}'
            }
        
    # get vehicle by deviceId from 'vehicles' collection
    def get_vehicle_by_device_id(self, device_id: str) -> Optional[Dict]:
        """
        Retrieve vehicle information by device ID from Firestore.
        
        Args:
            device_id (str): Device ID to search for
        """
        try:
            self._ensure_db_initialized()
            vehicles_ref = self.db.collection('vehicles')
            query = vehicles_ref.where(
                filter=FieldFilter('deviceId', '==', device_id)
            )
            results = query.stream()
            
            for doc in results:
                logger.info(f"Vehicle found for device ID {device_id}: {doc.id}")
                return doc
            
            logger.info(f"No vehicle found for device ID {device_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving vehicle by device ID: {e}")
            return None

# Create a singleton instance
firestore_helper = FirestoreHelper()