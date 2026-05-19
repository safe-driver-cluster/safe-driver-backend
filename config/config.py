import cv2

# System Configuration
VERSION_NO = "1.0.0"

# Language Settings
# 'en' for English
# 'si' for Sinhala (සිංහල)
# 'ta' for Tamil (தமிழ்)
LANGUAGE = 'en'

# Drowsiness Detection Thresholds
EYE_CLOSED_THRESH = 0.60
EYE_PARTIAL_THRESH = 0.40
MICROSLEEP_SEC = 1.5
PERCLOS_WIN_SEC = 60.0
PERCLOS_DROWSY = 0.20
EYE_CLOSURE_FREQ_WIN = 15.0
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

THRESHOLD_PHONE_COUNT = 3
THRESHOLD_BOTTLE_COUNT = 3
THRESHOLD_CIGARETTE_COUNT = 3

THRESHOLD_PHONE_ALERT_TO_CLOUD = 5
THRESHOLD_BOTTLE_ALERT_TO_CLOUD = 5
THRESHOLD_CIGARETTE_ALERT_TO_CLOUD = 3

VOICE_ALERT_PHONE = "Mobile phone use detected! Please focus on driving."
VOICE_ALERT_DRINKING = "Drinking detected! Please be careful when drinking while driving."
VOICE_ALERT_SMOKING = "Smoking detected! Please avoid smoking while driving."

VOICE_ALERT_PHONE_L2 = "Multiple mobile phone use events detected! Please focus on driving and minimize distractions."
VOICE_ALERT_DRINKING_L2 = "Multiple drinking events detected! If you are drowsy, please consider taking a break before continuing to drive."
VOICE_ALERT_SMOKING_L2 = "Multiple smoking events detected! Please avoid smoking while driving."

VOICE_ALERT_PHONE_L3 = "Frequent mobile phone use detected! I have to inform authorities if you continue to drive in this condition."
VOICE_ALERT_DRINKING_L3 = "Frequent drinking detected! I have to inform authorities if you continue to drive in this condition."
VOICE_ALERT_SMOKING_L3 = "Frequent smoking detected! I have to inform authorities if you continue to drive in this condition."

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
LABEL_CLOSURES = 'Closures(15s): {}'
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