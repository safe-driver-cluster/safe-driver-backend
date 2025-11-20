# safe-driver

### install python version 3.10.9
        https://www.python.org/downloads/release/python-3109/

### install python extension to vs-code
        Python by Microsoft microsoft.com

### check python version [3.10.9]
        python --version

## check the environment variables to ensure the path is correct

### create python environment
        python -m venv venv
        .\venv\Scripts\Activate

## - Temporarily Change Execution Policy
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

## - Permenantly Change Execution Policy
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

## --------------------------------------------------------------------------

### Upgrade pip (Important)
        pip install --upgrade pip
        python.exe -m pip install --upgrade pip

### install (or update) the project libraries
        pip install -r requirements.txt

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
        pip freeze > requirements.txt ##[DO NOT RUN THIS]

## -----------------------------------
## RUN SAFE DRIVER BACKEND APPLICATION
## -----------------------------------

# Run the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000

## 🧪 How to Use

# Save configurations:
curl -X POST http://localhost:8000/config/save

# Retrieve configurations:
curl -X GET http://localhost:8000/config/get

# Update a specific config:
curl -X PUT "http://localhost:8000/config/update?config_category=drowsiness_detection_thresholds&config_name=EYE_CLOSED_THRESH&config_value=0.65"

