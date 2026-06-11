
# ---------------------------------------------------
# SAFE DRIVER MONITORING SYSTEM - BACKEND GUIDELINES
# ---------------------------------------------------

The Raspberry Pi hardware API and cloud-hosted central API are separate
applications. See `CENTRAL_API_DEPLOYMENT.md` for central API deployment.

### install python version 3.10.9
        https://www.python.org/downloads/release/python-3109/

### install python extension to vs-code
        Python by Microsoft microsoft.com

### check python version [3.10.9]
        python --version

## check the environment variables to ensure the path is correct

### create python environment
        python -m venv venv
        py -3.11 -m venv venv           - using python 3.11.9
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
        logger.info('Sensor connected successfully!')
    else:
        logger.info('Wrong password!')
except Exception as e:
    print('Error:', e)
````

# --------------------------------
# GPS SIMULATOR
# --------------------------------

Use the simulator on Linux/Raspberry Pi to send the configured `GPS_SIGNALS`
from `gps/gps_simulator.py` to the backend as real NMEA GPS signals.

### Terminal 1 - Start the GPS simulator

```bash
cd ~/Desktop/safe-driver-backend
source venv/bin/activate
python gps/gps_simulator.py --pty --interval 5 --repeat
```

Keep Terminal 1 open. The simulator creates `/tmp/safe_driver_gps` and sends
the configured route signals repeatedly, with a five-second interval. Press
`Ctrl+C` when the simulation is finished.

### Terminal 2 - Start the backend normally

```bash
cd ~/Desktop/safe-driver-backend
source venv/bin/activate
python run.py
```

When `/tmp/safe_driver_gps` exists, the system automatically uses the
simulator instead of the real `/dev/ttyAMA5` GPS. Start the simulator before
starting `run.py`.

After stopping the simulator, `/tmp/safe_driver_gps` is removed. The next
normal system startup automatically uses the real GPS again.

### Useful simulator commands

Send signals every second:

```bash
python gps/gps_simulator.py --pty --interval 1
```

Repeat all configured GPS signals continuously:

```bash
python gps/gps_simulator.py --pty --interval 5 --repeat
```

Print generated NMEA signals without connecting to the backend:

```bash
python gps/gps_simulator.py --dry-run
```


## -------------------------------
## BUILD APPLICATION
## -------------------------------

Step 1 — Install PyInstaller in your venv
pip install pyinstaller

Step 2 - Use the existing `run.py` launcher.

Step 3 - Build the executable

Run each command from the project root after activating the correct platform's
virtual environment. PyInstaller builds must run on the target operating system
and architecture.

### Windows x64 build

This build packages the PyTorch `.pt` object-detection models.

```powershell
pyinstaller --clean --noconfirm --onefile --name safedriverapp-windows `
  --add-data "model/face_landmarker.task;model" `
  --add-data "model/yolov8n.pt;model" `
  --add-data "model/cigarette_model.pt;model" `
  --add-data "model/glasses_model.pt;model" `
  --add-data ".env;." `
  --add-data "banner.txt;." `
  --add-data "firebase-admin-sdk;firebase-admin-sdk" `
  --hidden-import database.storage_helper `
  --hidden-import gps.gps `
  --collect-all mediapipe `
  --collect-all ultralytics `
  run.py
```

Output: `dist/safedriverapp-windows.exe`

### Raspberry Pi OS ARM64 build

This build packages the ARM-friendly NCNN model directories. Do not package
the `.pt` models for Raspberry Pi object detection.

```bash
pyinstaller --clean --noconfirm --onefile --name safedriverapp-raspi-arm64 \
  --add-data "model/face_landmarker.task:model" \
  --add-data "model/yolov8n_ncnn_model:model/yolov8n_ncnn_model" \
  --add-data "model/cigarette_model_ncnn_model:model/cigarette_model_ncnn_model" \
  --add-data "model/glasses_model_ncnn_model:model/glasses_model_ncnn_model" \
  --add-data ".env:." \
  --add-data "banner.txt:." \
  --add-data "firebase-admin-sdk:firebase-admin-sdk" \
  --hidden-import database.storage_helper \
  --hidden-import gps.gps \
  --hidden-import fingerprint.enroll \
  --hidden-import fingerprint.remove \
  --hidden-import fingerprint.live \
  --collect-all mediapipe \
  --collect-all ultralytics \
  --collect-all ncnn \
  run.py
```

Output: `dist/safedriverapp-raspi-arm64`

The `audio/` directory is not packaged because voice-alert files are downloaded
from Firebase Storage during application startup.

Do not publish a build artifact containing
`firebase-admin-sdk/serviceAccountKey.json` to a public repository or release.
