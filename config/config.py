import cv2

# System Configuration
VERSION_NO = "1.0.0"

# Drowsiness Detection Thresholds
EYE_CLOSED_THRESH = 0.60
EYE_PARTIAL_THRESH = 0.40
MICROSLEEP_SEC = 1.5
PERCLOS_WIN_SEC = 60.0
PERCLOS_DROWSY = 0.70
YAWN_THRESH = 0.80
YAWN_MIN_SEC = 1.0
EYE_CLOSURE_FREQ_WIN = 15.0
EYE_CLOSURE_FREQ_THRESH = 4
MIN_CLOSURE_DURATION = 0.4
BLINK_MAX_DURATION = 0.4
CLOSURE_DEBOUNCE_TIME = 0.5

# Head Pose Detection Thresholds
HEAD_YAW_THRESH_LEFT = 35  # Degrees - threshold for left turn
HEAD_YAW_THRESH_RIGHT = 35  # Degrees - threshold for right turn
HEAD_TURN_DISTRACTION_SEC = 3.0  # Seconds - how long head must be turned to trigger alert
SHOW_HEAD_POSE_DETAILS = False  # Display head pose angles on screen

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

# FPS Display
FPS_FONT = cv2.FONT_HERSHEY_DUPLEX
FPS_FONT_SIZE = 0.5
FPS_FONT_THICKNESS = 1
FPS_COLOR = (0, 0, 0)
FPS_TEXT_FORMAT = 'FPS = {:.1f}'
FPS_Y_OFFSET = -20

# Metrics Box (Top Left)
METRICS_PADDING = 10
METRICS_WIDTH = 175
METRICS_HEIGHT = 165
METRICS_Y_OFFSET = 0
METRICS_CORNER_RADIUS = 10
METRICS_BG_COLOR = (255, 255, 255)
METRICS_BG_OPACITY = 0.5
METRICS_FONT = cv2.FONT_HERSHEY_SIMPLEX
METRICS_FONT_SIZE = 0.5
METRICS_FONT_THICKNESS = 1
METRICS_TEXT_COLOR = (0, 0, 0)

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
WARNING_COLOR = (0, 0, 255)
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

# Behavior Data Message Types
BEHAVIOR_FREQUENT_CLOSURES = 'frequent_closures'
BEHAVIOR_MICROSLEEP = 'microsleep'
BEHAVIOR_YAWN = 'yawn'
BEHAVIOR_DROWSY = 'drowsy'
BEHAVIOR_PERCLOS_REACHED = 'perclos_threshold_reached'
BEHAVIOR_DISTRACTION = 'distraction'
BEHAVIOR_MOBILE_USE = 'mobile_use'
BEHAVIOR_SMOKING = 'smoking'
BEHAVIOR_HEAD_TURN = "head_turn"

# Blendshapes Display
BLENDSHAPE_FONT = cv2.FONT_HERSHEY_SIMPLEX
BLENDSHAPE_FONT_SIZE = 0.4
BLENDSHAPE_FONT_THICKNESS = 1
BLENDSHAPE_TEXT_COLOR = (0, 0, 0)
BLENDSHAPE_BAR_COLOR = (0, 255, 0)
BLENDSHAPE_BAR_HEIGHT = 8
BLENDSHAPE_GAP_BETWEEN_BARS = 5
BLENDSHAPE_TEXT_GAP = 5
BLENDSHAPE_X_OFFSET = 20
BLENDSHAPE_Y_START = 30
BLENDSHAPE_TEXT_FORMAT = '{} ({:.2f})'

# Face Mesh Drawing Colors
LABEL_BG_COLOR = (255, 255, 255)

# Head Pose Display Settings
HEAD_POSE_DETAILS_Y_OFFSET = 240  # Y offset for head pose display
HEAD_POSE_FONT_SIZE = 0.5  # Font size for head pose text
HEAD_POSE_COLOR = (0, 0, 255)  # Color for head pose text (BGR)

# Camera Error Message
CAMERA_ERROR_MSG = 'ERROR: Unable to read from webcam. Please verify your webcam settings.'