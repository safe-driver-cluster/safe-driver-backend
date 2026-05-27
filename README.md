
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
        print('Sensor connected successfully!')
    else:
        print('Wrong password!')
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

pyinstaller --onefile --name safedriverapp --add-data "model/face_landmarker.task;model" --add-data "model/yolov8n.pt;model" --add-data "model/cigarette_model.pt;model" --add-data "model/glasses_model.pt;model" --add-data ".env;." --add-data "banner.txt;." --add-data "firebase-admin-sdk;firebase-admin-sdk" run.py

RASPBARRY
pyinstaller --onefile --name safedriverapp --add-data "model/face_landmarker.task:model" --add-data "model/yolov8n.pt:model" --add-data "model/cigarette_model.pt:model" --add-data "model/glasses_model.pt:model" --add-data ".env:." --add-data "banner.txt:." --add-data "firebase-admin-sdk:firebase-admin-sdk" run.py