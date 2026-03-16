"""
Hand Instrument — Computer Vision MIDI Controller
Built on top of github.com/frkatona/HandTrack_To_MIDI

RIGHT hand: play notes (wrist Y → pitch, pinch → trigger, finger spread → velocity)
LEFT hand:  expression (wrist Y → pitch bend, index Y → mod wheel)

Uses MediaPipe Tasks API (0.10.20+).
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import mido
from mido import Message
import sys
import os

# ═══════════════════════════════════════════════════════════════════
# CONFIG — edit these to taste
# ═══════════════════════════════════════════════════════════════════
MIDI_PORT = "HandMIDI"              # LoopMIDI virtual port name
SCALE = [0, 3, 5, 7, 10]           # pentatonic minor intervals
NOTE_MIN = 48                       # C3 (MIDI note number)
NOTE_MAX = 84                       # C6
VELOCITY_MIN = 40
VELOCITY_MAX = 127
PINCH_THRESHOLD = 0.05              # thumb-index distance to trigger note
MIDI_CHANNEL = 0                    # MIDI channel (0-15)
CAMERA_INDEX = 0                    # webcam index
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
# ═══════════════════════════════════════════════════════════════════

# Landmark indices (MediaPipe hand model)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
PINKY_TIP = 20

# Hand connection pairs for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle  (extra: 5-9)
    (0,13),(13,14),(14,15),(15,16),# ring    (extra: 9-13)
    (0,17),(17,18),(18,19),(19,20),# pinky   (extra: 13-17)
    (5,9),(9,13),(13,17),          # palm cross-connections
]

# Build the full list of scale notes in range
def build_scale_notes(scale, lo, hi):
    notes = []
    for midi_note in range(lo, hi + 1):
        if (midi_note - lo) % 12 in scale:
            notes.append(midi_note)
    return notes

SCALE_NOTES = build_scale_notes(SCALE, NOTE_MIN, NOTE_MAX)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_name(note):
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"

# ── MIDI setup ──────────────────────────────────────────────────────
available = mido.get_output_names()
midi_out = None
for name in available:
    if MIDI_PORT in name:
        midi_out = mido.open_output(name)
        print(f"Opened MIDI port: {name}")
        break

if midi_out is None:
    print(f"\n  ERROR: MIDI port '{MIDI_PORT}' not found.\n")
    print("  Available ports:")
    for p in available:
        print(f"    - {p}")
    print(f"\n  Create a virtual port named '{MIDI_PORT}' in LoopMIDI and try again.\n")
    sys.exit(1)

# ── MediaPipe HandLandmarker setup (Tasks API) ─────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"\n  ERROR: Model file not found at: {MODEL_PATH}")
    print("  Download it with:")
    print("  curl -sL -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n")
    midi_out.close()
    sys.exit(1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
detector = vision.HandLandmarker.create_from_options(options)

# ── State ───────────────────────────────────────────────────────────
active_note = None
is_pinching = False

hud_ready_note = "---"
hud_playing_note = "---"
hud_velocity = 0
hud_pitch_bend = 0
hud_cc1 = 0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def send_note_off(note):
    global active_note
    if note is not None:
        midi_out.send(Message('note_off', note=note, velocity=0, channel=MIDI_CHANNEL))
    active_note = None

def send_note_on(note, velocity):
    global active_note
    active_note = note
    midi_out.send(Message('note_on', note=note, velocity=velocity, channel=MIDI_CHANNEL))

def y_to_scale_note(y):
    """Map normalized Y (0=top, 1=bottom) to nearest scale note."""
    idx = int(np.interp(y, [0.0, 1.0], [len(SCALE_NOTES) - 1, 0]))
    idx = clamp(idx, 0, len(SCALE_NOTES) - 1)
    return SCALE_NOTES[idx]

def get_finger_spread(landmarks):
    """Distance between index tip and pinky tip."""
    idx = landmarks[INDEX_TIP]
    pinky = landmarks[PINKY_TIP]
    return np.sqrt((idx.x - pinky.x) ** 2 + (idx.y - pinky.y) ** 2)

def get_thumb_index_dist(landmarks):
    """Distance between thumb tip and index tip."""
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    return np.sqrt((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2)

def draw_hand_landmarks(frame, landmarks, color, conn_color):
    """Draw landmarks and connections on frame."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # draw connections
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], conn_color, 2)

    # draw landmark dots
    for pt in points:
        cv2.circle(frame, pt, 4, color, -1)
        cv2.circle(frame, pt, 4, (255, 255, 255), 1)

def draw_pitch_bar(frame, y_norm):
    """Vertical bar on right side showing pitch position."""
    h, w = frame.shape[:2]
    bar_x = w - 40
    bar_w = 20
    bar_top = 60
    bar_bot = h - 60

    # background
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (150, 150, 150), 1)

    # scale degree ticks
    for i in range(len(SCALE_NOTES)):
        tick_y = int(np.interp(i, [0, len(SCALE_NOTES) - 1], [bar_bot, bar_top]))
        cv2.line(frame, (bar_x - 3, tick_y), (bar_x, tick_y), (100, 100, 100), 1)

    # current position indicator
    pos_y = int(np.interp(y_norm, [0.0, 1.0], [bar_top, bar_bot]))
    cv2.rectangle(frame, (bar_x, pos_y - 3), (bar_x + bar_w, pos_y + 3), (0, 200, 255), -1)

    cv2.putText(frame, "PITCH", (bar_x - 10, bar_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

def draw_hud(frame):
    """Top-left HUD with current values."""
    lines = [
        f"READY:   {hud_ready_note}",
        f"PLAYING: {hud_playing_note}",
        f"VEL:     {hud_velocity}",
        f"BEND:    {hud_pitch_bend:+d}",
        f"MOD:     {hud_cc1}",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1, cv2.LINE_AA)
        y += 28

def draw_pinch_indicator(frame, wrist_lm, pinching):
    """Small circle near right wrist — gray idle, green pinching."""
    h, w = frame.shape[:2]
    cx = int(wrist_lm.x * w) + 30
    cy = int(wrist_lm.y * h) - 30
    color = (0, 255, 0) if pinching else (120, 120, 120)
    cv2.circle(frame, (cx, cy), 12, color, -1)
    cv2.circle(frame, (cx, cy), 12, (200, 200, 200), 1)

# ── Main loop ───────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    midi_out.close()
    sys.exit(1)

print("Hand Instrument running. Press 'q' to quit.")

frame_timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    h, w = frame.shape[:2]

    # convert to MediaPipe Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # detect hands
    frame_timestamp += 33  # ~30fps increment
    result = detector.detect_for_video(mp_image, frame_timestamp)

    right_found = False
    left_found = False

    if result.hand_landmarks:
        for hand_idx, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[hand_idx]
            label = handedness[0].category_name  # "Left" or "Right"

            # MediaPipe labels from camera perspective;
            # since we mirror, "Left" label = user's right hand
            is_right = (label == "Left")
            is_left = (label == "Right")

            wrist = landmarks[WRIST]

            # ── RIGHT HAND: melody ──────────────────────────────
            if is_right:
                right_found = True

                # draw in blue
                draw_hand_landmarks(frame, landmarks,
                                    color=(255, 150, 50), conn_color=(255, 100, 30))

                # target note from wrist Y
                note = y_to_scale_note(wrist.y)
                hud_ready_note = midi_to_name(note)

                # velocity from finger spread
                spread = get_finger_spread(landmarks)
                vel = int(np.interp(spread, [0.05, 0.25], [VELOCITY_MIN, VELOCITY_MAX]))
                vel = clamp(vel, VELOCITY_MIN, VELOCITY_MAX)
                hud_velocity = vel

                # pinch detection
                dist = get_thumb_index_dist(landmarks)
                pinching_now = dist < PINCH_THRESHOLD

                if pinching_now:
                    if not is_pinching:
                        send_note_off(active_note)
                        send_note_on(note, vel)
                        is_pinching = True
                        hud_playing_note = midi_to_name(note)
                    else:
                        if active_note != note:
                            send_note_off(active_note)
                            send_note_on(note, vel)
                            hud_playing_note = midi_to_name(note)
                else:
                    if is_pinching:
                        send_note_off(active_note)
                        is_pinching = False
                        hud_playing_note = "---"

                draw_pinch_indicator(frame, wrist, is_pinching)
                draw_pitch_bar(frame, wrist.y)

            # ── LEFT HAND: expression ───────────────────────────
            if is_left:
                left_found = True

                # draw in red
                draw_hand_landmarks(frame, landmarks,
                                    color=(50, 50, 255), conn_color=(30, 30, 200))

                # pitch bend from wrist Y
                bend = int(np.interp(wrist.y, [0.0, 1.0], [8191, -8191]))
                bend = clamp(bend, -8191, 8191)
                midi_out.send(Message('pitchwheel', pitch=bend, channel=MIDI_CHANNEL))
                hud_pitch_bend = bend

                # mod wheel (CC1) from index finger Y
                index_y = landmarks[INDEX_TIP].y
                cc1 = int(np.interp(index_y, [0.0, 1.0], [127, 0]))
                cc1 = clamp(cc1, 0, 127)
                midi_out.send(Message('control_change', control=1, value=cc1, channel=MIDI_CHANNEL))
                hud_cc1 = cc1

    # if right hand disappeared mid-note, kill it
    if not right_found and active_note is not None:
        send_note_off(active_note)
        is_pinching = False
        hud_playing_note = "---"
        hud_ready_note = "---"

    # "NO HANDS" overlay
    if not right_found and not left_found:
        text = "NO HANDS DETECTED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    draw_hud(frame)
    cv2.imshow("Hand Instrument", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ─────────────────────────────────────────────────────────
print("Shutting down...")
send_note_off(active_note)
cap.release()
cv2.destroyAllWindows()
midi_out.close()
print("Done.")
