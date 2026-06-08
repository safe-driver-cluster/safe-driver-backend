import cv2

# System Configuration
VERSION_NO = "1.0.0"

# Language Settings
# 'en' for English
# 'si' for Sinhala (සිංහල)
# 'ta' for Tamil (தமிழ்)
LANGUAGE = 'SINHALA'
SYSTEM = 'linux' # 'windows' or 'linux'

# Drowsiness Detection Thresholds
EYE_CLOSED_THRESH = 0.60
EYE_PARTIAL_THRESH = 0.40
MICROSLEEP_SEC = 1.5
PERCLOS_WIN_SEC = 60.0
PERCLOS_DROWSY = 0.50
EYE_CLOSURE_FREQ_WIN = 25.0
EYE_CLOSURE_FREQ_THRESH = 4
MIN_CLOSURE_DURATION = 0.4
BLINK_MAX_DURATION = 0.4
CLOSURE_DEBOUNCE_TIME = 0.5

#Yawning Detection Thresholds
YAWN_THRESH = 0.60
YAWN_MIN_SEC = 1.0

# Head Pose Detection Thresholds
HEAD_YAW_THRESH_LEFT = 35  # Degrees - threshold for left turn
HEAD_YAW_THRESH_RIGHT = 35  # Degrees - threshold for right turn
HEAD_TURN_DISTRACTION_SEC = 3.0  # Seconds - how long head must be turned to trigger alert
SHOW_HEAD_POSE_DETAILS = False  # Display head pose angles on screen

# Cloud Alert Level Thresholds
DROWSY_EVENT_COUNT_THRESH = 5
YAWN_EVENT_COUNT_THRESH = 5
MICROSLEEP_EVENT_COUNT_THRESH = 5
FREQUENT_CLOSURES_THRESH = 5
HEAD_TURN_COUNT_THRESH = 5
FACE_MISSING_COUNT_THRESH = 5

# Reset counters after this many seconds of no events (to prevent stale data from triggering alerts)
EVENT_COUNT_RESET_SEC = 300.0
EVENT_ARRAY_TIME_WINDOW_SEC = 600.0  # Time window to consider for counting events (e.g., count events in the last 10 minutes)

# Voice Alert Thresholds
VOICE_ALERT_COOLDOWN_SEC = 10.0  # Minimum seconds between voice alerts of the same type
VOICE_ALERT_CONSECUTIVE_EVENT_THRESH = 3  # Number of consecutive events to trigger voice alert
BUZZER_ALERT_COOLDOWN_SEC = 5.0  # Minimum seconds between buzzer alerts of the same type)
BUZZER_ALERT_CONSECUTIVE_EVENT_THRESH = 3  # Number of consecutive events to trigger buzzer alert
MAXIMUM_BUZZER_ALERTS_PER_TYPE = 2  # Maximum number of buzzer alerts per type to prevent spamming (first 3 alerts will be buzzered)
MAXIMUM_VOICE_ALERTS_PER_TYPE = 3  # Maximum number of voice alerts per type to prevent spamming (next 2 alerts will be voiced after buzzer limit is reached)

# Vibration Motor Alert Settings
ENABLE_VIBRATION_ALERTS = True
VIBRATION_GPIO_PIN = 18  # BCM GPIO18, connected to the vibration motor module IN pin.
VIBRATION_LEVEL_1 = 1
VIBRATION_LEVEL_2 = 2
VIBRATION_LEVEL_3 = 3
VIBRATION_LEVEL_4 = 4
VIBRATION_PATTERNS = {
    VIBRATION_LEVEL_1: (0.3, 1.5),   # Gentle pulse: 300 ms ON, 1500 ms OFF.
    VIBRATION_LEVEL_2: (0.5, 0.5),   # Moderate pulse: 500 ms ON, 500 ms OFF.
    VIBRATION_LEVEL_3: (1.0, 0.3),   # Strong pulse: 1000 ms ON, 300 ms OFF.
    VIBRATION_LEVEL_4: (None, None), # Emergency: continuous ON until stopped.
}
VIBRATION_WORKER_JOIN_TIMEOUT_SEC = 1.0

# -------------------------------------------------------------------------------------
# Object Detection Settings
# -------------------------------------------------------------------------------------

ENABLE_OBJECT_DETECTION = True  # Set to False to disable object detection (for performance testing)

YOLO_MODEL_PHONE_BOTTLE_PERSON_CONFIDENCE_THRESHOLD = 0.50
YOLO_MODEL_CIGARETTE_CONFIDENCE_THRESHOLD = 0.75
YOLO_MODEL_GLASSES_CONFIDENCE_THRESHOLD = 0.70

ENABLE_PHONE_BOTTLE_PERSON_DETECTION = True
ENABLE_CIGARETTE_DETECTION = True
ENABLE_GLASSES_DETECTION = False

ENABLE_CV2_WINDOW = False  # Set to False to disable cv2.imshow (for headless environments)
ENABLE_LOGGING = True  # Set to False to disable logging (for performance testing)

DETECT_PHONE_BOTTLE_PERSON_FRAME = 1
DETECT_CIGARETTE_FRAME = 1
DETECT_GLASSES_FRAME = 7

# Keep object detection lightweight on Raspberry Pi. YOLO and MediaPipe running
# at full rate together can exhaust memory or cause the Linux OOM killer to
# terminate the application without a Python traceback.
OBJECT_DETECTION_FRAME_INTERVAL = 3
OBJECT_DETECTION_FRAME_INTERVAL_LINUX = 12
OBJECT_DETECTION_IMGSZ = 416
OBJECT_DETECTION_IMGSZ_LINUX = 320
OBJECT_DETECTION_TORCH_THREADS_LINUX = 1

# Raspberry Pi object detection backend. PyTorch .pt inference can terminate
# with SIGILL on incompatible ARM wheels, so Linux uses exported NCNN models.
OBJECT_DETECTION_ALLOW_PYTORCH_FALLBACK_LINUX = False
YOLO_MODEL_PHONE_BOTTLE_PERSON = "model/yolov8n.pt"
YOLO_MODEL_CIGARETTE = "model/cigarette_model.pt"
YOLO_MODEL_GLASSES = "model/glasses_model.pt"
YOLO_MODEL_PHONE_BOTTLE_PERSON_LINUX = "model/yolov8n_ncnn_model"
YOLO_MODEL_CIGARETTE_LINUX = "model/cigarette_model_ncnn_model"
YOLO_MODEL_GLASSES_LINUX = "model/glasses_model_ncnn_model"

THRESHOLD_PHONE_COUNT = 3
THRESHOLD_BOTTLE_COUNT = 3
THRESHOLD_CIGARETTE_COUNT = 3

THRESHOLD_PHONE_ALERT_TO_CLOUD = 5
THRESHOLD_BOTTLE_ALERT_TO_CLOUD = 5
THRESHOLD_CIGARETTE_ALERT_TO_CLOUD = 3

# --------------------------------------------------------------------------------------

# UI Layout Parameters
WINDOW_NAME = 'SafeDriver Monitoring System'
ROW_SIZE = 50
LEFT_MARGIN = 24
LABEL_PADDING_WIDTH = 1500
FPS_AVG_FRAME_COUNT = 10
SCROLL_STEP = 20

# Display Control Flags
SHOW_BLENDSHAPES = False
SHOW_FACE_MESH = True
SHOW_FPS = True
SHOW_METRICS = True
SHOW_WARNINGS = True

# Voice Alert Control
ENABLE_VOICE_ALERTS = True

# FPS Display
FPS_FONT = cv2.FONT_HERSHEY_DUPLEX
FPS_FONT_SIZE = 0.5
FPS_FONT_THICKNESS = 1
FPS_COLOR = [0, 0, 0]
FPS_TEXT_FORMAT = 'FPS = {:.1f}'
FPS_Y_OFFSET = -20

# Metrics Box (Top Left)
METRICS_PADDING = 10
METRICS_WIDTH = 175
METRICS_HEIGHT = 165
METRICS_Y_OFFSET = 0
METRICS_CORNER_RADIUS = 10
METRICS_BG_COLOR = [255, 255, 255]
METRICS_BG_OPACITY = 0.5
METRICS_FONT = cv2.FONT_HERSHEY_SIMPLEX
METRICS_FONT_SIZE = 0.5
METRICS_FONT_THICKNESS = 1
METRICS_TEXT_COLOR = [0, 0, 0]

# Metrics Text Labels
LABEL_PERCLOS = 'PERCLOS: {:.2f}'
LABEL_BLINKS = 'Blinks/min: {:02d}'
LABEL_CLOSURES = 'Closures({}s): {}'
LABEL_YAWNS = 'Yawns: {}'
LABEL_MICROSLEEPS = 'Microsleeps: {}'
LABEL_DROWSY_EVENTS = 'Drowsy Events: {}'
LABEL_HEAD_POSE = 'Head Turn: {}'

# Metrics Text Positions
PERCLOS_Y_OFFSET = 20
BLINKS_Y_OFFSET = 40
CLOSURES_Y_OFFSET = 60
YAWNS_Y_OFFSET = 85
MICROSLEEPS_Y_OFFSET = 105
DROWSY_EVENTS_Y_OFFSET = 125
HEAD_POSE_Y_OFFSET = 150

# Warning Display
WARNING_FONT = cv2.FONT_HERSHEY_DUPLEX
WARNING_FONT_SIZE = 1.0
WARNING_FONT_THICKNESS = 2
WARNING_COLOR = [0, 0, 255]
WARNING_Y_POSITION = 50
WARNING_RIGHT_MARGIN = 20

# Warning Text Messages
WARNING_MICROSLEEP = 'Microsleep Detected!'
WARNING_YAWNING = 'Yawning Detected!'
WARNING_FREQUENT_CLOSURES = 'Frequent Eye Closures!'
WARNING_DROWSY = 'Drowsiness Detected!'
WARNING_PERCLOS = 'High PERCLOS Level!'
WARNING_DISTRACTION = 'Driver Distraction Detected!'
WARNING_MOBILE_USE = 'Mobile Phone Use Detected!'
WARNING_SMOKING = 'Smoking Detected!'
WARNING_HEAD_TURN = "Head Turn Detected!"
WARNING_DRINKING = "Drinking Detected!"

# Console Messages
CONSOLE_MICROSLEEP = 'Microsleep detected (Total: {})'
CONSOLE_YAWN = 'Yawn detected (Total: {})'
CONSOLE_FREQUENT_CLOSURES = 'Frequent eye closures detected'
CONSOLE_DROWSY = 'Drowsiness detected (Total: {})'
CONSOLE_PERCLOS_REACHED = 'PERCLOS threshold reached: {:.2f}'
CONSOLE_DISTRACTION = 'Driver distraction detected'
CONSOLE_MOBILE_USE = 'Mobile phone use detected'
CONSOLE_SMOKING = 'Smoking detected'
CONSOLE_HEAD_TURN = "Head turned {} for {:.2f}s"
CONSOLE_FACE_LOSS = "Driver face not visible for {:.2f}s - Complete turn away detected"
CONSOLE_DRINKING = "Drinking detected"

# Behavior Data Message Types
BEHAVIOR_FREQUENT_CLOSURES = 'frequent_closures'
BEHAVIOR_MICROSLEEP = 'microsleep'
BEHAVIOR_YAWN = 'yawn'
BEHAVIOR_DROWSY = 'drowsy'
BEHAVIOR_PERCLOS_REACHED = 'perclos_threshold_reached'
BEHAVIOR_DISTRACTION = 'distraction'
BEHAVIOR_HEAD_TURN = "head_turn"
BEHAVIOR_MOBILE_USE = 'mobile_use'
BEHAVIOR_SMOKING = 'smoking'
BEHAVIOR_DRINKING = 'drinking'

VOICE_ALERT_DEFAULT = "Alert detected! Please stay focused on driving."

# Blendshapes Display
BLENDSHAPE_FONT = cv2.FONT_HERSHEY_SIMPLEX
BLENDSHAPE_FONT_SIZE = 0.4
BLENDSHAPE_FONT_THICKNESS = 1
BLENDSHAPE_TEXT_COLOR = [0, 0, 0]
BLENDSHAPE_BAR_COLOR = [0, 255, 0]
BLENDSHAPE_BAR_HEIGHT = 8
BLENDSHAPE_GAP_BETWEEN_BARS = 5
BLENDSHAPE_TEXT_GAP = 5
BLENDSHAPE_X_OFFSET = 20
BLENDSHAPE_Y_START = 30
BLENDSHAPE_TEXT_FORMAT = '{} ({:.2f})'

# Face Mesh Drawing Colors
LABEL_BG_COLOR = [255, 255, 255]

# Head Pose Display Settings
HEAD_POSE_DETAILS_Y_OFFSET = 240  # Y offset for head pose display
HEAD_POSE_FONT_SIZE = 0.5  # Font size for head pose text
HEAD_POSE_COLOR = [0, 0, 255]  # Color for head pose text (BGR)

# Camera Error Message
CAMERA_ERROR_MSG = 'ERROR: Unable to read from webcam. Please verify your webcam settings.'

ENABLE_WINDOW = True  # Set to False to disable cv2.imshow (for headless environments)
ENABLE_DETECTION = True  # Set to False to disable all detection (for performance testing)
ENABLE_FINGERPRINT = False  # Set to False to disable fingerprinting (for performance testing)
ENABLE_GPS = False  # Set to False to disable GPS location retrieval (for performance testing)

# ======================================================
# GPS CONFIGURATIONS
# ======================================================

DEVICE_ID = ""
SPEED_LIMIT = 20.0
SPEED_THRESHOLD = 2.0
PUSH_INTERVAL = 5          # push every 5 seconds minimum
SPEED_CHANGE_THRESHOLD = 3 # push immediately if speed changes by 3 km/h
CURRENT_SPEED = 0.0

# =================================================================================================================================
# MULTILINGUAL CONFIGURATIONS
# =================================================================================================================================

VOICE_ALERT_PHONE = "Mobile phone use detected! Please focus on driving."
VOICE_ALERT_DRINKING = "Drinking detected! Please be careful when drinking while driving."
VOICE_ALERT_SMOKING = "Smoking detected! Please avoid smoking while driving."

VOICE_ALERT_PHONE_L2 = "Multiple mobile phone use events detected! Please focus on driving and minimize distractions."
VOICE_ALERT_DRINKING_L2 = "Multiple drinking events detected! If you are drowsy, please consider taking a break before continuing to drive."
VOICE_ALERT_SMOKING_L2 = "Multiple smoking events detected! Please avoid smoking while driving."

VOICE_ALERT_PHONE_L3 = "Frequent mobile phone use detected! I have to inform authorities if you continue to drive in this condition."
VOICE_ALERT_DRINKING_L3 = "Frequent drinking detected! I have to inform authorities if you continue to drive in this condition."
VOICE_ALERT_SMOKING_L3 = "Frequent smoking detected! I have to inform authorities if you continue to drive in this condition."

# # -- ENGLISH MESSAGES --

# VOICE_ALERT_PHONE_ENGLISH = "Mobile phone use detected! Please focus on driving."
# VOICE_ALERT_DRINKING_ENGLISH = "Drinking detected! Please be careful when drinking while driving."
# VOICE_ALERT_SMOKING_ENGLISH = "Smoking detected! Please avoid smoking while driving."

# VOICE_ALERT_PHONE_L2_ENGLISH = "Multiple mobile phone use events detected! Please focus on driving and minimize distractions."
# VOICE_ALERT_DRINKING_L2_ENGLISH = "Multiple drinking events detected! If you are drowsy, please consider taking a break before continuing to drive."
# VOICE_ALERT_SMOKING_L2_ENGLISH = "Multiple smoking events detected! Please avoid smoking while driving."

# VOICE_ALERT_PHONE_L3_ENGLISH = "Frequent mobile phone use detected! I have to inform authorities if you continue to drive in this condition."
# VOICE_ALERT_DRINKING_L3_ENGLISH = "Frequent drinking detected! I have to inform authorities if you continue to drive in this condition."
# VOICE_ALERT_SMOKING_L3_ENGLISH = "Frequent smoking detected! I have to inform authorities if you continue to drive in this condition."

# # -- SINHALA TRANSLATIONS --

# VOICE_ALERT_PHONE_SINHALA = "ජංගම දුරකථන භාවිතයක් හඳුනාගෙන ඇත! කරුණාකර රිය පැදවීමට අවධානය යොමු කරන්න."
# VOICE_ALERT_DRINKING_SINHALA = "බීම වර්ගයක් පානය කිරීම හඳුනාගෙන ඇත! රිය පදවන අතරතුර බීම වර්ග පානය කිරීමේදී කරුණාකර සැලකිලිමත් වන්න."
# VOICE_ALERT_SMOKING_SINHALA = "දුම්පානය කිරීමක් හඳුනාගෙන ඇත! කරුණාකර රිය පදවන අතරතුර දුම්පානයෙන් වැළකී සිටින්න."

# VOICE_ALERT_PHONE_L2_SINHALA = "ජංගම දුරකථනය කිහිප වරක් භාවිත කර ඇති බව හඳුනාගෙන ඇත! කරුණාකර වෙනත් බාහිර දේවලින් මිදී රිය පැදවීමට අවධානය යොමු කරන්න."
# VOICE_ALERT_DRINKING_L2_SINHALA = "කිහිප වරක් බීම වර්ග පානය කර ඇති බව හඳුනාගෙන ඇත! ඔබට නිදිමත ගතියක් දැනේ නම්, නැවත රිය පැදවීමට පෙර සුළු විවේකයක් ගැනීමට සලකා බලන්න."
# VOICE_ALERT_SMOKING_L2_SINHALA = "කිහිප වරක් දුම්පානය කර ඇති බව හඳුනාගෙන ඇත! කරුණාකර රිය පදවන අතරතුර දුම්පානයෙන් වැළකී සිටින්න."

# VOICE_ALERT_PHONE_L3_SINHALA = "නිතර නිතර ජංගම දුරකථනය භාවිත කරන බව හඳුනාගෙන ඇත! ඔබ දිගින් දිගටම මෙලෙස රිය පැදවුවහොත් මට බලධාරීන් දැනුවත් කිරීමට සිදුවේ."
# VOICE_ALERT_DRINKING_L3_SINHALA = "නිතර නිතර බීම වර්ග පානය කරන බව හඳුනාගෙන ඇත! ඔබ දිගින් දිගටම මෙලෙස රිය පැදවුවහොත් මට බලධාරීන් දැනුවත් කිරීමට සිදුවේ."
# VOICE_ALERT_SMOKING_L3_SINHALA = "නිතර නිතර දුම්පානය කරන බව හඳුනාගෙන ඇත! ඔබ දිගින් දිගටම මෙලෙස රිය පැදවුවහොත් මට බලධාරීන් දැනුවත් කිරීමට සිදුවේ."

# # -- TAMIL TRANSLATIONS --

# VOICE_ALERT_PHONE_TAMIL = "கைபேசி பயன்பாடு கண்டறியப்பட்டுள்ளது! தயவுசெய்து வாகனத்தை ஓட்டுவதில் கவனம் செலுத்துங்கள்."
# VOICE_ALERT_DRINKING_TAMIL = "பானம் அருந்துவது கண்டறியப்பட்டுள்ளது! வாகனம் ஓட்டும்போது பானங்கள் அருந்துவதில் கவனமாக இருக்கவும்."
# VOICE_ALERT_SMOKING_TAMIL = "புகைபிடித்தல் கண்டறியப்பட்டுள்ளது! வாகனம் ஓட்டும்போது புகைபிடிப்பதைத் தவிர்க்கவும்."

# VOICE_ALERT_PHONE_L2_TAMIL = "தொடர்ச்சியான கைபேசி பயன்பாடு கண்டறியப்பட்டுள்ளது! தயவுசெய்து கவனச்சிதறல்களைக் குறைத்து, வாகனம் ஓட்டுவதில் கவனம் செலுத்துங்கள்."
# VOICE_ALERT_DRINKING_L2_TAMIL = "தொடர்ச்சியாக பானம் அருந்துவது கண்டறியப்பட்டுள்ளது! உங்களுக்கு தூக்கக் கலக்கம் இருந்தால், தொடர்ந்து வாகனம் ஓட்டுவதற்கு முன் சிறிது ஓய்வெடுக்கவும்."
# VOICE_ALERT_SMOKING_L2_TAMIL = "தொடர்ச்சியான புகைபிடித்தல் கண்டறியப்பட்டுள்ளது! வாகனம் ஓட்டும்போது புகைபிடிப்பதைத் தவிர்க்கவும்."

# VOICE_ALERT_PHONE_L3_TAMIL = "அடிக்கடி கைபேசி பயன்படுத்துவது கண்டறியப்பட்டுள்ளது! நீங்கள் இதே நிலையில் தொடர்ந்து வாகனம் ஓட்டினால், நான் அதிகாரிகளுக்குத் தகவல் தெரிவிக்க வேண்டியிருக்கும்."
# VOICE_ALERT_DRINKING_L3_TAMIL = "அடிக்கடி பானம் அருந்துவது கண்டறியப்பட்டுள்ளது! நீங்கள் இதே நிலையில் தொடர்ந்து வாகனம் ஓட்டினால், நான் அதிகாரிகளுக்குத் தகவல் தெரிவிக்க வேண்டியிருக்கும்."
# VOICE_ALERT_SMOKING_L3_TAMIL = "அடிக்கடி புகைபிடிப்பது கண்டறியப்பட்டுள்ளது! நீங்கள் இதே நிலையில் தொடர்ந்து வாகனம் ஓட்டினால், நான் அதிகாரிகளுக்குத் தகவல் தெரிவிக்க வேண்டியிருக்கும்."

# =================================================================================================================================

# Voice Alert Messages
VOICE_ALERT_MICROSLEEP = 'Microsleep detected! Please stay alert.'
VOICE_ALERT_YAWNING = 'Yawning detected! Please stay focused.'
VOICE_ALERT_DROWSY = 'Drowsiness detected! Please take a break.'
VOICE_ALERT_DISTRACTION = 'Driver distraction detected! Please pay attention to the road.'
VOICE_ALERT_HEAD_TURN = 'Head turn detected! Please keep your eyes on the road.'
VOICE_ALERT_PERCLOS = 'High PERCLOS level detected! Please stay alert.'
VOICE_ALERT_FREQUENT_CLOSURES = 'Frequent eye closures detected! Please stay alert.'

# Voice Alert Messages Level 2 (after buzzer limit is reached)
VOICE_ALERT_MICROSLEEP_L2 = 'Multiple microsleep events detected! Please take a break.'
VOICE_ALERT_YAWNING_L2 = 'Multiple yawning events detected! Please take a break.'
VOICE_ALERT_DROWSY_L2 = 'Multiple drowsiness events detected! Please take a break.'
VOICE_ALERT_DISTRACTION_L2 = 'Multiple distraction events detected! Please focus on driving.'
VOICE_ALERT_HEAD_TURN_L2 = 'Multiple head turn events detected! Please keep your eyes on the road.'
VOICE_ALERT_PERCLOS_L2 = 'PERCLOS level has been high multiple times! Please stay alert and consider taking a break.'
VOICE_ALERT_FREQUENT_CLOSURES_L2 = 'Frequent eye closures detected! Please stay alert and consider taking a break.'

# Voice Alert Messages Level 3 (warning to driver by mentioning have to inform authorities)
VOICE_ALERT_MICROSLEEP_L3 = 'Frequent microsleep events detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
VOICE_ALERT_YAWNING_L3 = 'Frequent yawning events detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
VOICE_ALERT_DROWSY_L3 = 'Frequent drowsiness events detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
VOICE_ALERT_DISTRACTION_L3 = 'Frequent distraction events detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
VOICE_ALERT_HEAD_TURN_L3 = 'Frequent head turn events detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
VOICE_ALERT_PERCLOS_L3 = 'PERCLOS level has been high frequently! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
VOICE_ALERT_FREQUENT_CLOSURES_L3 = 'Frequent eye closures detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'


# # -- ENGLISH MESSAGES --

# # Voice Alert Messages
# VOICE_ALERT_MICROSLEEP_ENGLISH = 'Microsleep detected! Please stay alert.'
# VOICE_ALERT_YAWNING_ENGLISH = 'Yawning detected! Please stay focused.'
# VOICE_ALERT_DROWSY_ENGLISH = 'Drowsiness detected! Please take a break.'
# VOICE_ALERT_DISTRACTION_ENGLISH = 'Driver distraction detected! Please pay attention to the road.'
# VOICE_ALERT_HEAD_TURN_ENGLISH = 'Head turn detected! Please keep your eyes on the road.'
# VOICE_ALERT_PERCLOS_ENGLISH = 'High PERCLOS level detected! Please stay alert.'
# VOICE_ALERT_FREQUENT_CLOSURES_ENGLISH = 'Frequent eye closures detected! Please stay alert.'

# # Voice Alert Messages Level 2 (after buzzer limit is reached)
# VOICE_ALERT_MICROSLEEP_L2_ENGLISH = 'Multiple microsleep events detected! Please take a break.'
# VOICE_ALERT_YAWNING_L2_ENGLISH = 'Multiple yawning events detected! Please take a break.'
# VOICE_ALERT_DROWSY_L2_ENGLISH = 'Multiple drowsiness events detected! Please take a break.'
# VOICE_ALERT_DISTRACTION_L2_ENGLISH = 'Multiple distraction events detected! Please focus on driving.'
# VOICE_ALERT_HEAD_TURN_L2_ENGLISH = 'Multiple head turn events detected! Please keep your eyes on the road.'
# VOICE_ALERT_PERCLOS_L2_ENGLISH = 'PERCLOS level has been high multiple times! Please stay alert and consider taking a break.'
# VOICE_ALERT_FREQUENT_CLOSURES_L2_ENGLISH = 'Frequent eye closures detected! Please stay alert and consider taking a break.'

# # Voice Alert Messages Level 3 (warning to driver by mentioning have to inform authorities)
# VOICE_ALERT_MICROSLEEP_L3_ENGLISH = 'Frequent microsleep events detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
# VOICE_ALERT_YAWNING_L3_ENGLISH = 'Frequent yawning events detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
# VOICE_ALERT_DROWSY_L3_ENGLISH = 'Frequent drowsiness events detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
# VOICE_ALERT_DISTRACTION_L3_ENGLISH = 'Frequent distraction events detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
# VOICE_ALERT_HEAD_TURN_L3_ENGLISH = 'Frequent head turn events detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
# VOICE_ALERT_PERCLOS_L3_ENGLISH = 'PERCLOS level has been high frequently! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'
# VOICE_ALERT_FREQUENT_CLOSURES_L3_ENGLISH = 'Frequent eye closures detected! If you continue to drive in this condition, authorities may be notified for your safety and the safety of others on the road.'

# # -- SINHALA TRANSLATIONS --

# # Voice Alert Messages
# VOICE_ALERT_MICROSLEEP_SINHALA = 'සුළු මොහොතක නින්දක් (Microsleep) හඳුනාගෙන ඇත! කරුණාකර අවධානයෙන් සිටින්න.'
# VOICE_ALERT_YAWNING_SINHALA = 'ඈනුම් යෑමක් හඳුනාගෙන ඇත! කරුණාකර අවධානය යොමු කරන්න.'
# VOICE_ALERT_DROWSY_SINHALA = 'නිදිමත ගතියක් හඳුනාගෙන ඇත! කරුණාකර සුළු විවේකයක් ගන්න.'
# VOICE_ALERT_DISTRACTION_SINHALA = 'රියදුරුගේ අවධානය ගිලිහී ඇති බව හඳුනාගෙන ඇත! කරුණාකර මාර්ගය වෙත අවධානය යොමු කරන්න.'
# VOICE_ALERT_HEAD_TURN_SINHALA = 'හිස හරවා බැලීමක් හඳුනාගෙන ඇත! කරුණාකර ඔබේ දෑස් මාර්ගය වෙතම යොමු කර තබාගන්න.'
# VOICE_ALERT_PERCLOS_SINHALA = 'ඉහළ PERCLOS මට්ටමක් හඳුනාගෙන ඇත! කරුණාකර අවධානයෙන් සිටින්න.'
# VOICE_ALERT_FREQUENT_CLOSURES_SINHALA = 'නිතර නිතර දෑස් වැසී යන බව හඳුනාගෙන ඇත! කරුණාකර අවධානයෙන් සිටින්න.'

# # Voice Alert Messages Level 2
# VOICE_ALERT_MICROSLEEP_L2_SINHALA = 'සුළු මොහොතක නින්දයාමේ අවස්ථා කිහිපයක් හඳුනාගෙන ඇත! කරුණාකර සුළු විවේකයක් ගන්න.'
# VOICE_ALERT_YAWNING_L2_SINHALA = 'ඈනුම් ඇරීමේ අවස්ථා කිහිපයක් හඳුනාගෙන ඇත! කරුණාකර සුළු විවේකයක් ගන්න.'
# VOICE_ALERT_DROWSY_L2_SINHALA = 'නිදිමත ගතියේ අවස්ථා කිහිපයක් හඳුනාගෙන ඇත! කරුණාකර සුළු විවේකයක් ගන්න.'
# VOICE_ALERT_DISTRACTION_L2_SINHALA = 'අවධානය ගිලිහී යාමේ අවස්ථා කිහිපයක් හඳුනාගෙන ඇත! කරුණාකර රිය පැදවීමට අවධානය යොමු කරන්න.'
# VOICE_ALERT_HEAD_TURN_L2_SINHALA = 'හිස හරවා බැලීමේ අවස්ථා කිහිපයක් හඳුනාගෙන ඇත! කරුණාකර ඔබේ දෑස් මාර්ගය වෙතම යොමු කර තබාගන්න.'
# VOICE_ALERT_PERCLOS_L2_SINHALA = 'PERCLOS මට්ටම කිහිප වරක්ම ඉහළ ගොස් ඇත! කරුණාකර අවධානයෙන් සිටින්න, නවාතැනක් ගෙන විවේක ගැනීමට සලකා බලන්න.'
# VOICE_ALERT_FREQUENT_CLOSURES_L2_SINHALA = 'නිතර නිතර දෑස් වැසී යන බව හඳුනාගෙන ඇත! කරුණාකර අවධානයෙන් සිටින්න, නවාතැනක් ගෙන විවේක ගැනීමට සලකා බලන්න.'

# # Voice Alert Messages Level 3
# VOICE_ALERT_MICROSLEEP_L3_SINHALA = 'නිතර නිතර සුළු මොහොතක නින්දයාමේ අවස්ථා හඳුනාගෙන ඇත! ඔබ දිගින් දිගටම මෙම තත්ත්වයෙන් රිය පැදවුවහොත්, ඔබගේ සහ අන් අයගේ ආරක්ෂාව වෙනුවෙන් බලධාරීන් දැනුවත් කිරීමට සිදුවිය හැක.'
# VOICE_ALERT_YAWNING_L3_SINHALA = 'නිතර නිතර ඈනුම් ඇරීමේ අවස්ථා හඳුනාගෙන ඇත! ඔබ දිගින් දිගටම මෙම තත්ත්වයෙන් රිය පැදවුවහොත්, ඔබගේ සහ අන් අයගේ ආරක්ෂාව වෙනුවෙන් බලධාරීන් දැනුවත් කිරීමට සිදුවිය හැක.'
# VOICE_ALERT_DROWSY_L3_SINHALA = 'නිතර නිතර නිදිමත ගතියේ අවස්ථා හඳුනාගෙන ඇත! ඔබ දිගින් දිගටම මෙම තත්ත්වයෙන් රිය පැදවුවහොත්, ඔබගේ සහ අන් අයගේ ආරක්ෂාව වෙනුවෙන් බලධාරීන් දැනුවත් කිරීමට සිදුවිය හැක.'
# VOICE_ALERT_DISTRACTION_L3_SINHALA = 'නිතර නිතර අවධානය ගිලිහී යාමේ අවස්ථා හඳුනාගෙන ඇත! ඔබ දිගින් දිගටම මෙම තත්ත්වයෙන් රිය පැදවුවහොත්, ඔබගේ සහ අන් අයගේ ආරක්ෂාව වෙනුවෙන් බලධාරීන් දැනුවත් කිරීමට සිදුවිය හැක.'
# VOICE_ALERT_HEAD_TURN_L3_SINHALA = 'නිතර නිතර හිස හරවා බැලීමේ අවස්ථා හඳුනාගෙන ඇත! ඔබ දිගින් දිගටම මෙම තත්ත්වයෙන් රිය පැදවුවහොත්, ඔබගේ සහ අන් අයගේ ආරක්ෂාව වෙනුවෙන් බලධාරීන් දැනුවත් කිරීමට සිදුවිය හැක.'
# VOICE_ALERT_PERCLOS_L3_SINHALA = 'PERCLOS මට්ටම නිතර නිතර ඉහළ ගොස් ඇත! ඔබ දිගින් දිගටම මෙම තත්ත්වයෙන් රිය පැදවුවහොත්, ඔබගේ සහ අන් අයගේ ආරක්ෂාව වෙනුවෙන් බලධාරීන් දැනුවත් කිරීමට සිදුවිය හැක.'
# VOICE_ALERT_FREQUENT_CLOSURES_L3_SINHALA = 'නිතර නිතර දෑස් වැසී යන බව හඳුනාගෙන ඇත! ඔබ දිගින් දිගටම මෙම තත්ත්වයෙන් රිය පැදවුවහොත්, ඔබගේ සහ අන් අයගේ ආරක්ෂාව වෙනුවෙන් බලධාරීන් දැනුවත් කිරීමට සිදුවිය හැක.'

# # -- TAMIL TRANSLATIONS --

# # Voice Alert Messages
# VOICE_ALERT_MICROSLEEP_TAMIL = 'குறுகிய நேரத் தூக்கம் (Microsleep) கண்டறியப்பட்டுள்ளது! தயவுசெய்து விழிப்புடன் இருக்கவும்.'
# VOICE_ALERT_YAWNING_TAMIL = 'கொட்டாவி விடுவது கண்டறியப்பட்டுள்ளது! தயவுசெய்து கவனமாக இருக்கவும்.'
# VOICE_ALERT_DROWSY_TAMIL = 'சோர்வு/தூக்கக் கலக்கம் கண்டறியப்பட்டுள்ளது! தயவுசெய்து சிறிது ஓய்வெடுக்கவும்.'
# VOICE_ALERT_DISTRACTION_TAMIL = 'ஓட்டுநரின் கவனச்சிதறல் கண்டறியப்பட்டுள்ளது! தயவுசெய்து சாலையில் கவனம் செலுத்தவும்.'
# VOICE_ALERT_HEAD_TURN_TAMIL = 'தலை திருப்புவது கண்டறியப்பட்டுள்ளது! தயவுசெய்து பார்வையைச் சாலையின் மீது வைக்கவும்.'
# VOICE_ALERT_PERCLOS_TAMIL = 'அதிகப்படியான PERCLOS அளவு கண்டறியப்பட்டுள்ளது! தயவுசெய்து விழிப்புடன் இருக்கவும்.'
# VOICE_ALERT_FREQUENT_CLOSURES_TAMIL = 'அடிக்கடி கண்கள் மூடுவது கண்டறியப்பட்டுள்ளது! தயவுசெய்து விழிப்புடன் இருக்கவும்.'

# # Voice Alert Messages Level 2
# VOICE_ALERT_MICROSLEEP_L2_TAMIL = 'தொடர்ச்சியான குறுகிய நேரத் தூக்கம் கண்டறியப்பட்டுள்ளது! தயவுசெய்து சிறிது ஓய்வெடுக்கவும்.'
# VOICE_ALERT_YAWNING_L2_TAMIL = 'தொடர்ச்சியாக கொட்டாவி விடுவது கண்டறியப்பட்டுள்ளது! தயவுசெய்து சிறிது ஓய்வெடுக்கவும்.'
# VOICE_ALERT_DROWSY_L2_TAMIL = 'தொடர்ச்சியான தூக்கக் கலக்கம் கண்டறியப்பட்டுள்ளது! தயவுசெய்து சிறிது ஓய்வெடுக்கவும்.'
# VOICE_ALERT_DISTRACTION_L2_TAMIL = 'தொடர்ச்சியான கவனச்சிதறல் கண்டறியப்பட்டுள்ளது! தயவுசெய்து வாகனம் ஓட்டுவதில் கவனம் செலுத்தவும்.'
# VOICE_ALERT_HEAD_TURN_L2_TAMIL = 'தொடர்ச்சியாக தலை திருப்புவது கண்டறியப்பட்டுள்ளது! தயவுசெய்து பார்வையைச் சாலையின் மீது வைக்கவும்.'
# VOICE_ALERT_PERCLOS_L2_TAMIL = 'PERCLOS அளவு பலமுறை அதிகமாக இருந்துள்ளது! தயவுசெய்து விழிப்புடன் இருந்து, சிறிது ஓய்வெடுக்கவும்.'
# VOICE_ALERT_FREQUENT_CLOSURES_L2_TAMIL = 'அடிக்கடி கண்கள் மூடுவது கண்டறியப்பட்டுள்ளது! தயவுசெய்து விழிப்புடன் இருந்து, சிறிது ஓய்வெடுக்கவும்.'

# # Voice Alert Messages Level 3
# VOICE_ALERT_MICROSLEEP_L3_TAMIL = 'அடிக்கடி குறுகிய நேரத் தூக்கம் கண்டறியப்பட்டுள்ளது! நீங்கள் இதே நிலையில் தொடர்ந்து வாகனம் ஓட்டினால், உங்கள் பாதுகாப்பிற்காகவும் மற்றவர்களின் பாதுகாப்பிற்காகவும் அதிகாரிகளுக்குத் தகவல் தெரிவிக்கப்படலாம்.'
# VOICE_ALERT_YAWNING_L3_TAMIL = 'அடிக்கடி கொட்டாவி விடுவது கண்டறியப்பட்டுள்ளது! நீங்கள் இதே நிலையில் தொடர்ந்து வாகனம் ஓட்டினால், உங்கள் பாதுகாப்பிற்காகவும் மற்றவர்களின் பாதுகாப்பிற்காகவும் அதிகாரிகளுக்குத் தகவல் தெரிவிக்கப்படலாம்.'
# VOICE_ALERT_DROWSY_L3_TAMIL = 'அடிக்கடி தூக்கக் கலக்கம் கண்டறியப்பட்டுள்ளது! நீங்கள் இதே நிலையில் தொடர்ந்து வாகனம் ஓட்டினால், உங்கள் பாதுகாப்பிற்காகவும் மற்றவர்களின் பாதுகாப்பிற்காகவும் அதிகாரிகளுக்குத் தகவல் தெரிவிக்கப்படலாம்.'
# VOICE_ALERT_DISTRACTION_L3_TAMIL = 'அடிக்கடி கவனச்சிதறல் கண்டறியப்பட்டுள்ளது! நீங்கள் இதே நிலையில் தொடர்ந்து வாகனம் ஓட்டினால், உங்கள் பாதுகாப்பிற்காகவும் மற்றவர்களின் பாதுகாப்பிற்காகவும் அதிகாரிகளுக்குத் தகவல் தெரிவிக்கப்படலாம்.'
# VOICE_ALERT_HEAD_TURN_L3_TAMIL = 'அடிக்கடி தலை திருப்புவது கண்டறியப்பட்டுள்ளது! நீங்கள் இதே நிலையில் தொடர்ந்து வாகனம் ஓட்டினால், உங்கள் பாதுகாப்பிற்காகவும் மற்றவர்களின் பாதுகாப்பிற்காகவும் அதிகாரிகளுக்குத் தகவல் தெரிவிக்கப்படலாம்.'
# VOICE_ALERT_PERCLOS_L3_TAMIL = 'PERCLOS அளவு அடிக்கடி அதிகமாக இருந்துள்ளது! நீங்கள் இதே நிலையில் தொடர்ந்து வாகனம் ஓட்டினால், உங்கள் பாதுகாப்பிற்காகவும் மற்றவர்களின் பாதுகாப்பிற்காகவும் அதிகாரிகளுக்குத் தகவல் தெரிவிக்கப்படலாம்.'
# VOICE_ALERT_FREQUENT_CLOSURES_L3_TAMIL = 'அடிக்கடி கண்கள் மூடுவது கண்டறியப்பட்டுள்ளது! நீங்கள் இதே நிலையில் தொடர்ந்து வாகனம் ஓட்டினால், உங்கள் பாதுகாப்பிற்காகவும் மற்றவர்களின் பாதுகாப்பிற்காகவும் அதிகாரிகளுக்குத் தகவல் தெரிவிக்கப்படலாம்.'
