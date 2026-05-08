import os
import argparse
import firebase_admin
from firebase_admin import credentials

from database.firestore_helper import firestore_helper


def init_firebase():
    """Initialize Firebase Admin SDK once for this script."""
    admin_sdk_path = os.getenv(
        'ADMIN_SDK_PATH',
        '/home/safedriver/Desktop/safe-driver-backend/firebase-admin-sdk/serviceAccountKey.json',
    )

    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(admin_sdk_path)
        firebase_admin.initialize_app(cred)


def main():
    parser = argparse.ArgumentParser(
        description='Delete driver fingerprint mapping from Firestore.'
    )
    parser.add_argument('--driver-id', help='Driver ID in drivers collection')
    parser.add_argument('--template-id', help='Template ID from fingerprint_templates collection')
    parser.add_argument('--scanner-id', help='Scanner/device ID used to build template ID')
    parser.add_argument('--template-position', type=int, help='Template slot position inside scanner')
    parser.add_argument(
        '--keep-legacy-fingerprint-id',
        action='store_true',
        help='Do not remove legacy drivers/{driver_id}.fingerprint_id field',
    )

    args = parser.parse_args()

    if (
        args.driver_id is None
        and args.template_id is None
        and (args.scanner_id is None or args.template_position is None)
    ):
        parser.error('Provide --driver-id or --template-id or --scanner-id + --template-position')

    init_firebase()

    result = firestore_helper.delete_driver_fingerprint(
        driver_id=args.driver_id,
        scanner_id=args.scanner_id,
        template_position=args.template_position,
        template_id=args.template_id,
        delete_legacy_fingerprint_id=not args.keep_legacy_fingerprint_id,
    )

    print(result)


if __name__ == '__main__':
    main()
