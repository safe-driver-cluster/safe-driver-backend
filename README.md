
# ---------------------------------------------------
# SAFE DRIVER MONITORING SYSTEM - BACKEND GUIDELINES
# ---------------------------------------------------

### install python version 3.10.9
        https://www.python.org/downloads/release/python-3109/

### install python extension to vs-code
        Python by Microsoft microsoft.com

### check python version [3.10.9]
        python --version

## check the environment variables to ensure the path is correct

### create python environment
        python -m venv venv
        .\venv\Scripts\Activate         - WINDOWS

        python -m venv venv             - [use python 3.13.5 because there not exists dabian versions for all windows dependancies in python 3.10]
        source ./venv/bin/activate      - RASPBARRY

## - Temporarily Change Execution Policy
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

## - Permenantly Change Execution Policy
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

## --------------------------------------------------------------------------

### Upgrade pip (Important)
        pip install --upgrade pip
        python.exe -m pip install --upgrade pip

### install (or update) the project libraries
        pip install -r requirements.txt         (in windows)
        pip install -r requirements_raspi.txt   (in raspi)

## DEVELOPER MODE INSTRUCTIONS [DO NOT RUN BELLOW]

### download dlib library
        https://github.com/z-mahmud22/Dlib_Windows_Python3.x/blob/main/dlib-19.22.99-cp310-cp310-win_amd64.whl

### copy the dlib library to 
        path --> safe-driver-model/

## install libraries
        pip install mediapipe opencv-python numpy gps3
        pip install cmake
        pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
        pip install face_recognition

### save the working environment (run only when new libraries installed)
        pip freeze > requirements_n.txt

        ##[DO NOT RUN THIS COMMAND]
        pip freeze > requirements.txt

## rename the firebase-admin-sdk -> serviceAccountKey.json

## ------------------------------------
## HARDWARE ARCHITECTURE
## ------------------------------------

SafeDriver runs on a Raspberry Pi 4 with camera-based driver monitoring, GPS,
fingerprint authentication, audio buzzer alerts, text-to-speech voice alerts,
Firebase cloud reporting, and haptic feedback.

Hardware modules:

- Raspberry Pi 4 Model B
- Pi Camera Module v2 / NoIR Camera
- NEO-6M GPS Module
- AS608 Fingerprint Sensor
- Audio Buzzer
- 5V Vibration Motor Module

The 5V vibration motor module provides haptic feedback for driver awareness.
It is controlled by GPIO18 and operates alongside the buzzer and voice warning
channels.

## ------------------------------------
## RASPBERRY PI WIRING DIAGRAM
## ------------------------------------

Vibration Motor Module:

- GPIO18 -> IN
- 5V -> VCC
- GND -> GND

The module contains its own transistor driver. GPIO18 only sends HIGH and LOW
signals; no extra transistor control logic is required in software.

## ------------------------------------
## TECHNOLOGY STACK
## ------------------------------------

- OpenCV for camera frame processing
- MediaPipe for face landmarks and blendshapes
- TensorFlow Lite / YOLO for inference and object detection
- Firebase Realtime Database and Firestore for cloud alerts
- Text-to-speech for voice alerts
- GPIO buzzer for audible warnings
- 5V Vibration Motor Module for driver haptic alerting

## ------------------------------------
## ALERT ESCALATION SYSTEM
## ------------------------------------

SafeDriver uses a multi-level escalation model:

- Level 1: Gentle vibration warning only. No cloud notification or emergency escalation.
- Level 2: Vibration, buzzer, and voice warning.
- Level 3: Strong vibration, buzzer, voice warning, cloud notification, dashboard update, and event logging.
- Level 4: Continuous vibration, continuous buzzer, emergency voice warning, Firebase emergency alert, dashboard emergency notification, and SMS escalation if available.

Vibration patterns:

- Level 1: 300 ms ON, 1500 ms OFF, repeat.
- Level 2: 500 ms ON, 500 ms OFF, repeat.
- Level 3: 1000 ms ON, 300 ms OFF, repeat.
- Level 4: Continuous ON until the driver acknowledges the alert or the unsafe condition is cleared.

## ------------------------------------
## VIBRATION MOTOR IMPLEMENTATION
## ------------------------------------

The vibration motor is implemented as an independent service in
`service/vibration_service.py`. Detection code emits behavior events, the
central `AlertManager` decides the alert level, and the vibration service runs
the matching GPIO18 pattern in the background.

Resource management strategy:

- GPIO18 is configured as an output pin using BCM numbering.
- The service lazily initializes Raspberry Pi GPIO so development machines can run without GPIO hardware.
- Vibration patterns run on a daemon worker thread and never block camera capture, OpenCV, MediaPipe, TensorFlow, YOLO, or Firebase communication.
- Only one vibration pattern runs at a time. When the alert level changes, the previous pattern is stopped before the next one starts.
- Shutdown cleanup turns GPIO18 LOW and releases GPIO resources via `atexit`, including keyboard interrupts and normal application exits.

Driver safety benefit:

Haptic feedback provides an early physical cue even before audible or voice
warnings are needed. This is useful when a driver is starting to become drowsy,
is distracted, or when road noise reduces the effectiveness of audio alerts.

## ------------------------------------
## VIBRATION TESTING PROCEDURE
## ------------------------------------

1. Confirm wiring: GPIO18 -> IN, 5V -> VCC, GND -> GND.
2. Activate the Raspberry Pi virtual environment.
3. Run the normal backend test suite:

```
pytest
```

4. Start the application and trigger a drowsiness or distraction condition.
5. Confirm Level 1 starts gentle vibration immediately.
6. Confirm repeated events escalate to Level 2 and Level 3 patterns.
7. Stop the application and confirm the motor turns off.

## ------------------------------------
## RASPBERRY PI DEPLOYMENT PROCEDURE
## ------------------------------------

1. Install Raspberry Pi dependencies:

```
pip install -r requirements_raspi.txt
```

2. Ensure `RPi.GPIO` is available in the Raspberry Pi environment.
3. Wire the vibration motor module to GPIO18, 5V, and GND.
4. Start the backend:

```
uvicorn main:app --host 0.0.0.0 --port 8000
```

5. Verify that alert escalation activates haptic feedback without blocking video processing.

## ------------------------------------
## RUN SAFE DRIVER BACKEND APPLICATION
## ------------------------------------

# Run the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000

## -------------------------------
## 🧪 HOW TO USE API CALLS IN CMD
## -------------------------------

# Set default configurations:
curl -X POST http://localhost:8000/config/save

# Retrieve configurations:
curl -X GET http://localhost:8000/config/get

# Update a specific config:
curl -X PUT "http://localhost:8000/config/update?config_name=EYE_CLOSED_THRESH&config_value=0.65"

# Update a specific config and restart the detection process
curl -X PUT "http://localhost:8000/config/update-and-restart?config_name=ENABLE_VOICE_ALERTS&config_value=true"

# Restart the detection process
curl -X PUT "http://localhost:8000/process/restart"

# Check application running status
curl -X GET http://localhost:8000/process/status"


## ----------------------------------------
## LINUX COMMANDS
## ----------------------------------------

# Activate venv
source venv/bin/activate

### CONNECT TO RASPBERRY-PI

ssh -4 safedriver@raspberrypi.local
rensith2001

python3.10 -m venv venv

source ./venv/bin/activate

### RASPI CAMERA OPERATIONS

### FINGERPRINT OPERATION

🔌 1. Hardware Connection (VERY IMPORTANT)

⚙️ 2. Enable Serial Port on Raspberry Pi
Run: sudo raspi-config
Go to: Interface Options → Serial Port
Set: Enable serial port hardware → Yes

🔧 3. Disable Serial Console (if needed)
edit: sudo nano /boot/firmware/cmdline.txt
✂️ Edit the line and REMOVE ONLY this part:
console=serial0,115200
Then reboot: sudo reboot

📦 4. Install Required Python Library
pip install pyfingerprint

🧪 5. Test Connection
````
from pyfingerprint.pyfingerprint import PyFingerprint

try:
    f = PyFingerprint('/dev/serial0', 57600, 0xFFFFFFFF, 0x00000000)

    if f.verifyPassword():
        logger.info('Sensor connected successfully!')
    else:
        logger.info('Wrong password!')
except Exception as e:
    print('Error:', e)
````

## -------------------------------
## BUILD APPLICATION
## -------------------------------

Step 1 — Install PyInstaller in your venv
pip install pyinstaller

Step 2 — Create a launcher file [run.py]
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False   # reload must be OFF for compiled builds
    )

Step 3 — Build the executable

WINDOWS
pyinstaller --onefile --name safedriverapp --add-data "config;config" --add-data "model;model" --add-data "service;service" --add-data "utils;utils" --add-data ".env;." --add-data "banner.txt;." --add-data "shared.py;shared.py" --add-data "database;database" --add-data "firebase-admin-sdk;firebase-admin-sdk" run.py

pyinstaller --onefile --name safedriverapp --add-data "model/face_landmarker.task;model" --add-data "model/yolov8n.pt;model" --add-data "model/cigarette_model.pt;model" --add-data "model/glasses_model.pt;model" --add-data "service;service" --add-data ".env;." --add-data "banner.txt;." --add-data "firebase-admin-sdk;firebase-admin-sdk" --hidden-import=dotenv run.py

RASPBARRY
pyinstaller --onefile --name safedriverapp --add-data "model/face_landmarker.task:model" --add-data "model/yolov8n.pt:model" --add-data "model/cigarette_model.pt:model" --add-data "model/glasses_model.pt:model" --add-data "service:service" --add-data ".env:." --add-data "banner.txt:." --add-data "firebase-admin-sdk:firebase-admin-sdk" run.py
