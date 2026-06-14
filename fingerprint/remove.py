if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys
from fingerprint.utils import (
    announce,
    get_scanner_id,
    build_fingerprint_template_id,
    wait_for_finger,
    init_audio,
)
from fingerprint.sensor import create_sensor
from database.firestore_helper import FirestoreHelper
from service.model_service import (get_mac_address_alternative)

firestore_helper = FirestoreHelper()


def delete_fingerprint(
    position: int | None = None,
    driver_id: str | None = None,
    mac: str | None = None,
) -> bool:
    """
    Delete a fingerprint from both the sensor and Firestore.

    Provide at least one of:
      - position      : template slot index on the sensor
      - driver_id     : Firestore driver document ID
                        (template position is resolved from Firestore)

    Returns True on full success, False if any step failed.
    """
    device_mac = get_mac_address_alternative()
    if mac.upper() != device_mac.upper():
        announce(f"MAC address mismatch. Operation allowed only on device with MAC {device_mac}.")
        return False

    if driver_id is None:
        announce("Provide a template position or driver ID to delete.")
        return False

    scanner_id = get_scanner_id()

    # ------------------------------------------------------------------ #
    # 1. If only driver_id was given, resolve the template position first  #
    # ------------------------------------------------------------------ #
    if position is None:
        driver_data = firestore_helper.get_driver(driver_id)          # adjust to your actual getter
        if not driver_data:
            announce(f"Driver {driver_id} not found in database.")
            return False

        position = driver_data.get("fingerprint_template_position")
        scanner = driver_data.get("fingerprint_scanner_id")
        if position is None and scanner != scanner_id:
            announce("No fingerprint template position found for this driver for this device.")
            return False

    # ------------------------------------------------------------------ #
    # 2. Delete from sensor                                               #
    # ------------------------------------------------------------------ #
    sensor_deleted = False
    try:
        sensor = create_sensor()

        if sensor.deleteTemplate(position):
            sensor_deleted = True
            announce(f"Template at position {position} removed from sensor.")
        else:
            announce(f"Sensor reported failure deleting template at position {position}.")
            return False

    except Exception as e:
        announce(f"Sensor error: {e}")
        return False

    # ------------------------------------------------------------------ #
    # 3. Build template ID and delete from Firestore                      #
    # ------------------------------------------------------------------ #
    template_id = build_fingerprint_template_id(scanner_id, position)

    result = firestore_helper.delete_driver_fingerprint(
        driver_id=driver_id,
        scanner_id=scanner_id,
        template_position=position,
        template_id=template_id,
    )

    if result.get("success"):
        announce("Fingerprint deleted successfully.")
        return True
    else:
        # Sensor deletion already happened — log the inconsistency clearly
        msg = result.get("message", "Unknown error")
        announce(
            f"Sensor record removed but Firestore deletion failed: {msg}. "
            "Manual cleanup may be required."
        )
        return False


# ---------------------------------------------------------------------- #
# CLI entry point                                                          #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    init_audio()

    parser = argparse.ArgumentParser(description="Delete a registered fingerprint.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--position", type=int, help="Template slot index on the sensor")
    group.add_argument("--driver-id", dest="driver_id", help="Firestore driver document ID")
    args = parser.parse_args()

    success = delete_fingerprint(
        position=args.position,
        driver_id=args.driver_id,
    )
    sys.exit(0 if success else 1)
