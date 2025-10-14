import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("firebase-admin-sdk\safe-driver-system-firebase-adminsdk-fbsvc-76241499ba.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://safe-driver-system-default-rtdb.firebaseio.com/'  # Replace with your actual database URL
})

# Firestore connection
firestore_db = firebase_admin.firestore.client()

# Realtime Database connection
def get_realtime_db():
    """
    Returns a reference to the Firebase Realtime Database
    """
    return db.reference('/')  # Root reference, you can specify a path like '/users'

# Function to get a reference to a specific path in the Realtime Database
def get_realtime_db_ref(path):
    """
    Returns a reference to a specific path in the Firebase Realtime Database
    
    Args:
        path (str): The path to the data in the database
    
    Returns:
        db.Reference: A reference to the specified database path
    """
    return db.reference(path)