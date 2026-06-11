import json
import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, db


def initialize_firebase():
    try:
        return firebase_admin.get_app()
    except ValueError:
        pass

    database_url = os.environ["FIREBASE_DATABASE_URL"]
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if service_account_json:
        credential = credentials.Certificate(json.loads(service_account_json))
    elif service_account_path:
        credential = credentials.Certificate(Path(service_account_path))
    else:
        credential = credentials.ApplicationDefault()

    return firebase_admin.initialize_app(
        credential,
        {"databaseURL": database_url},
    )


def reference(path: str):
    initialize_firebase()
    return db.reference(path)
