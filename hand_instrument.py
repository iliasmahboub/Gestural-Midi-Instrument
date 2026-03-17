"""
Hand Instrument - Computer Vision MIDI Controller
Built on top of github.com/frkatona/HandTrack_To_MIDI

Right hand: play notes (wrist Y -> pitch, pinch -> trigger, finger spread -> velocity)
Left hand: expression (wrist Y -> pitch bend, index Y -> mod wheel)

Uses MediaPipe Tasks API (0.10.20+).
"""

import os
import sys

import cv2
import mediapipe as mp
import mido
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mido import Message

# =====================================================================
# CONFIG
# =====================================================================
MIDI_PORT = "HandMIDI"
MIDI_CHANNEL = 0
CAMERA_INDEX = 0
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task",
)

NOTE_MIN = 48
NOTE_MAX = 84
VELOCITY_MIN = 40
VELOCITY_MAX = 127
PINCH_ON_THRESHOLD = 0.045
PINCH_OFF_THRESHOLD = 0.065
BEND_SMOOTHING = 0.35
MOD_SMOOTHING = 0.25

ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALE_LIBRARY = [
    {"name": "Major", "family": "Bright", "intervals": [0, 2, 4, 5, 7, 9, 11]},
    {"name": "Minor Pentatonic", "family": "Blues", "intervals": [0, 3, 5, 7, 10]},
    {"name": "Major Pentatonic", "family": "Open", "intervals": [0, 2, 4, 7, 9]},
    {"name": "Major Bebop", "family": "Jazz", "intervals": [0, 2, 4, 5, 7, 8, 9, 11]},
    {"name": "Mixolydian", "family": "Loose", "intervals": [0, 2, 4, 5, 7, 9, 10]},
    {"name": "Lydian", "family": "Luminous", "intervals": [0, 2, 4, 6, 7, 9, 11]},
    {"name": "Natural Minor", "family": "Aeolian", "intervals": [0, 2, 3, 5, 7, 8, 10]},
    {"name": "Harmonic Minor", "family": "Dramatic", "intervals": [0, 2, 3, 5, 7, 8, 11]},
    {"name": "Melodic Minor", "family": "Fluid", "intervals": [0, 2, 3, 5, 7, 9, 11]},
    {"name": "Minor Bebop", "family": "Jazz", "intervals": [0, 2, 3, 5, 7, 8, 9, 10]},
    {"name": "Dorian", "family": "Soulful", "intervals": [0, 2, 3, 5, 7, 9, 10]},
    {"name": "Phrygian", "family": "Dark", "intervals": [0, 1, 3, 5, 7, 8, 10]},
    {"name": "Phrygian Dominant", "family": "Tense", "intervals": [0, 1, 4, 5, 7, 8, 10]},
    {"name": "Double Harmonic", "family": "Cinematic", "intervals": [0, 1, 4, 5, 7, 8, 11]},
    {"name": "Hijaz", "family": "Arabic", "intervals": [0, 1, 4, 5, 7, 8, 10]},
    {"name": "Bayati Approx", "family": "Arabic", "intervals": [0, 1, 3, 5, 7, 8, 10]},
    {"name": "Kurd", "family": "Arabic", "intervals": [0, 1, 3, 5, 7, 8, 10]},
    {"name": "Sika Approx", "family": "Arabic", "intervals": [0, 1, 4, 5, 7, 9, 10]},
    {"name": "Nahawand", "family": "Arabic", "intervals": [0, 2, 3, 5, 7, 8, 10]},
    {"name": "Whole Tone", "family": "Floating", "intervals": [0, 2, 4, 6, 8, 10]},
]

DEFAULT_SCALE_INDEX = 0
DEFAULT_ROOT_INDEX = 0

# Landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

NOTE_NAMES = ROOT_NAMES


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def midi_to_name(note):
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


def current_scale_label():
    scale = SCALE_LIBRARY[current_scale_index]
    return f"{ROOT_NAMES[current_root_index]} {scale['name']}"


def build_scale_notes(root_index, intervals, lo, hi):
    notes = []
    for midi_note in range(lo, hi + 1):
        if (midi_note - root_index) % 12 in intervals:
            notes.append(midi_note)
    return notes


def refresh_scale_notes():
    global scale_notes
    scale_notes = build_scale_notes(
        current_root_index,
        SCALE_LIBRARY[current_scale_index]["intervals"],
        NOTE_MIN,
        NOTE_MAX,
    )


def cycle_scale(direction):
    global current_scale_index
    current_scale_index = (current_scale_index + direction) % len(SCALE_LIBRARY)
    refresh_scale_notes()


def cycle_root(direction):
    global current_root_index
    current_root_index = (current_root_index + direction) % len(ROOT_NAMES)
    refresh_scale_notes()


def y_to_scale_note(y):
    idx = int(np.interp(y, [0.0, 1.0], [len(scale_notes) - 1, 0]))
    idx = clamp(idx, 0, len(scale_notes) - 1)
    return scale_notes[idx]


def get_finger_spread(landmarks):
    index = landmarks[INDEX_TIP]
    pinky = landmarks[PINKY_TIP]
    return float(np.hypot(index.x - pinky.x, index.y - pinky.y))


def get_thumb_index_dist(landmarks):
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    return float(np.hypot(thumb.x - index.x, thumb.y - index.y))


def velocity_from_spread(spread):
    normalized = np.interp(spread, [0.05, 0.25], [0.0, 1.0])
    normalized = float(clamp(normalized, 0.0, 1.0))
    curved = normalized ** 0.7
    return int(np.interp(curved, [0.0, 1.0], [VELOCITY_MIN, VELOCITY_MAX]))


def smooth_value(current, target, alpha):
    return current + (target - current) * alpha


def send_note_off(note):
    global active_note
    if note is not None:
        midi_out.send(Message("note_off", note=note, velocity=0, channel=MIDI_CHANNEL))
    active_note = None


def send_note_on(note, velocity):
    global active_note
    active_note = note
    midi_out.send(Message("note_on", note=note, velocity=velocity, channel=MIDI_CHANNEL))


def draw_hand_landmarks(frame, landmarks, color, conn_color):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], conn_color, 2)

    for pt in points:
        cv2.circle(frame, pt, 4, color, -1)
        cv2.circle(frame, pt, 4, (255, 255, 255), 1)


def draw_pitch_bar(frame, y_norm):
    h, w = frame.shape[:2]
    bar_x = w - 90
    bar_w = 22
    bar_top = 78
    bar_bot = h - 70

    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bot), (145, 145, 145), 1)

    tick_step = 1 if len(scale_notes) <= 16 else max(1, len(scale_notes) // 12)
    for i, note in enumerate(scale_notes):
        tick_y = int(np.interp(i, [0, len(scale_notes) - 1], [bar_bot, bar_top]))
        tick_color = (95, 95, 95)
        cv2.line(frame, (bar_x - 6, tick_y), (bar_x, tick_y), tick_color, 1)
        if i % tick_step == 0:
            cv2.putText(
                frame,
                midi_to_name(note),
                (bar_x - 66, tick_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (170, 170, 170),
                1,
                cv2.LINE_AA,
            )

    pos_y = int(np.interp(y_norm, [0.0, 1.0], [bar_top, bar_bot]))
    cv2.rectangle(frame, (bar_x - 2, pos_y - 5), (bar_x + bar_w + 2, pos_y + 5), (0, 210, 255), -1)
    cv2.putText(
        frame,
        "PITCH MAP",
        (bar_x - 34, bar_top - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )


def draw_hud(frame):
    scale = SCALE_LIBRARY[current_scale_index]
    lines = [
        f"SCALE:   {current_scale_label()}",
        f"COLOR:   {scale['family']}",
        f"READY:   {hud_ready_note}",
        f"PLAYING: {hud_playing_note}",
        f"VEL:     {hud_velocity}",
        f"BEND:    {hud_pitch_bend:+d}",
        f"MOD:     {hud_cc1}",
    ]
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        y += 26


def draw_pinch_indicator(frame, wrist_lm, pinching):
    h, w = frame.shape[:2]
    cx = int(wrist_lm.x * w) + 34
    cy = int(wrist_lm.y * h) - 34
    if pinching:
        cv2.circle(frame, (cx, cy), 22, (0, 120, 0), 2)
    color = (0, 255, 110) if pinching else (120, 120, 120)
    cv2.circle(frame, (cx, cy), 12, color, -1)
    cv2.circle(frame, (cx, cy), 12, (220, 220, 220), 1)


def draw_control_hints(frame):
    text = "[ and ] scale   , and . root   s picker   q quit"
    cv2.putText(
        frame,
        text,
        (16, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )


def draw_scale_picker(frame):
    if not show_scale_picker:
        return

    h, w = frame.shape[:2]
    panel_w = 460
    panel_h = 390
    x0 = (w - panel_w) // 2
    y0 = (h - panel_h) // 2
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (12, 18, 22), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 220, 180), 2)

    cv2.putText(frame, "Scale Picker", (x0 + 20, y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 215), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Root: {ROOT_NAMES[current_root_index]}    Family: {SCALE_LIBRARY[current_scale_index]['family']}",
        (x0 + 20, y0 + 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (215, 215, 215),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Arrow up/down chooses scale, left/right changes root, s closes",
        (x0 + 20, y1 - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    visible_count = 9
    half_window = visible_count // 2
    start = max(0, current_scale_index - half_window)
    end = min(len(SCALE_LIBRARY), start + visible_count)
    start = max(0, end - visible_count)

    line_y = y0 + 106
    for idx in range(start, end):
        scale = SCALE_LIBRARY[idx]
        previous_family = SCALE_LIBRARY[idx - 1]["family"] if idx > 0 else None
        if scale["family"] != previous_family:
            cv2.putText(
                frame,
                scale["family"].upper(),
                (x0 + 30, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (110, 210, 180),
                1,
                cv2.LINE_AA,
            )
            line_y += 20

        is_selected = idx == current_scale_index
        if is_selected:
            cv2.rectangle(frame, (x0 + 18, line_y - 18), (x1 - 18, line_y + 10), (18, 72, 66), -1)
        label = f"{ROOT_NAMES[current_root_index]} {scale['name']}"
        color = (255, 255, 255) if is_selected else (188, 188, 188)
        cv2.putText(frame, label, (x0 + 30, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)
        line_y += 28


def handle_picker_key(key):
    if key == 2490368:
        cycle_scale(-1)
    elif key == 2621440:
        cycle_scale(1)
    elif key == 2424832:
        cycle_root(-1)
    elif key == 2555904:
        cycle_root(1)


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
    for port_name in available:
        print(f"    - {port_name}")
    print(f"\n  Create a virtual port named '{MIDI_PORT}' in LoopMIDI and try again.\n")
    sys.exit(1)

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

current_scale_index = DEFAULT_SCALE_INDEX
current_root_index = DEFAULT_ROOT_INDEX
scale_notes = []
refresh_scale_notes()

active_note = None
is_pinching = False
show_scale_picker = False
smoothed_bend = 0.0
smoothed_cc1 = 0.0

hud_ready_note = "---"
hud_playing_note = "---"
hud_velocity = 0
hud_pitch_bend = 0
hud_cc1 = 0

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

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_timestamp += 33
    result = detector.detect_for_video(mp_image, frame_timestamp)

    right_found = False
    left_found = False

    if result.hand_landmarks:
        for hand_idx, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[hand_idx]
            label = handedness[0].category_name

            is_right = label == "Left"
            is_left = label == "Right"

            wrist = landmarks[WRIST]

            if is_right:
                right_found = True
                draw_hand_landmarks(frame, landmarks, color=(255, 150, 50), conn_color=(255, 100, 30))

                note = y_to_scale_note(wrist.y)
                hud_ready_note = midi_to_name(note)

                spread = get_finger_spread(landmarks)
                vel = velocity_from_spread(spread)
                hud_velocity = vel

                dist = get_thumb_index_dist(landmarks)
                if is_pinching:
                    pinching_now = dist < PINCH_OFF_THRESHOLD
                else:
                    pinching_now = dist < PINCH_ON_THRESHOLD

                if pinching_now:
                    if not is_pinching:
                        send_note_off(active_note)
                        send_note_on(note, vel)
                        is_pinching = True
                        hud_playing_note = midi_to_name(note)
                    elif active_note != note:
                        send_note_off(active_note)
                        send_note_on(note, vel)
                        hud_playing_note = midi_to_name(note)
                elif is_pinching:
                    send_note_off(active_note)
                    is_pinching = False
                    hud_playing_note = "---"

                draw_pinch_indicator(frame, wrist, is_pinching)
                draw_pitch_bar(frame, wrist.y)

            if is_left:
                left_found = True
                draw_hand_landmarks(frame, landmarks, color=(50, 50, 255), conn_color=(30, 30, 200))

                target_bend = int(np.interp(wrist.y, [0.0, 1.0], [8191, -8191]))
                smoothed_bend = smooth_value(smoothed_bend, target_bend, BEND_SMOOTHING)
                bend = clamp(int(round(smoothed_bend)), -8191, 8191)
                midi_out.send(Message("pitchwheel", pitch=bend, channel=MIDI_CHANNEL))
                hud_pitch_bend = bend

                index_y = landmarks[INDEX_TIP].y
                target_cc1 = int(np.interp(index_y, [0.0, 1.0], [127, 0]))
                smoothed_cc1 = smooth_value(smoothed_cc1, target_cc1, MOD_SMOOTHING)
                cc1 = clamp(int(round(smoothed_cc1)), 0, 127)
                midi_out.send(Message("control_change", control=1, value=cc1, channel=MIDI_CHANNEL))
                hud_cc1 = cc1

    if not right_found and active_note is not None:
        send_note_off(active_note)
        is_pinching = False
        hud_playing_note = "---"
        hud_ready_note = "---"

    if not right_found:
        hud_ready_note = "---"
        hud_velocity = 0

    if not left_found:
        smoothed_bend = smooth_value(smoothed_bend, 0, BEND_SMOOTHING)
        smoothed_cc1 = smooth_value(smoothed_cc1, 0, MOD_SMOOTHING)
        hud_pitch_bend = clamp(int(round(smoothed_bend)), -8191, 8191)
        hud_cc1 = clamp(int(round(smoothed_cc1)), 0, 127)
        midi_out.send(Message("pitchwheel", pitch=hud_pitch_bend, channel=MIDI_CHANNEL))
        midi_out.send(Message("control_change", control=1, value=hud_cc1, channel=MIDI_CHANNEL))

    if not right_found and not left_found:
        text = "NO HANDS DETECTED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    draw_hud(frame)
    draw_control_hints(frame)
    draw_scale_picker(frame)
    cv2.imshow("Hand Instrument", frame)

    key = cv2.waitKeyEx(1)
    if key == ord("q"):
        break
    if key == ord("s"):
        show_scale_picker = not show_scale_picker
    elif key == ord("["):
        cycle_scale(-1)
    elif key == ord("]"):
        cycle_scale(1)
    elif key == ord(","):
        cycle_root(-1)
    elif key == ord("."):
        cycle_root(1)
    elif show_scale_picker:
        handle_picker_key(key)

print("Shutting down...")
send_note_off(active_note)
cap.release()
cv2.destroyAllWindows()
midi_out.close()
print("Done.")
