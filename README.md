# Hand Instrument — Computer Vision MIDI Controller

Play MIDI notes and control expression in FL Studio using your hands and a webcam. Built on top of [HandTrack_To_MIDI](https://github.com/frkatona/HandTrack_To_MIDI).

## How It Works

| Hand | What it controls |
|------|-----------------|
| **Right hand** | **Melody** — Wrist height picks the note (C pentatonic minor, C3-C6). Pinch thumb + index to play. Finger spread controls velocity. Slide while pinching for legato. |
| **Left hand** | **Expression** — Wrist height sends pitch bend (-8191 to +8191). Index finger height sends mod wheel (CC1, 0-127). |

## Setup

### 1. Install LoopMIDI (virtual MIDI cable)
- Download from [Tobias Erichsen's site](https://www.tobias-erichsen.de/software/loopmidi.html)
- Open LoopMIDI, create a new port named **HandMIDI**
- Leave LoopMIDI running in the background

### 2. Configure FL Studio
- Go to **Options > MIDI Settings**
- Under **Input**, find **HandMIDI** and enable it
- Set its port number (e.g., port 0)
- Click the **Enable** button so FL receives from this port

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```
Requires Python 3.10-3.12 (MediaPipe does not yet support 3.13).

### 4. Run
```bash
python hand_instrument.py
```

## Mapping Controls in FL Studio

- **Notes** arrive on MIDI channel 1 — arm a channel/instrument to receive them
- **CC1 (mod wheel)** and **pitch bend** can be linked to any knob:
  1. Right-click any knob or slider in FL Studio
  2. Click **Link to controller...**
  3. Move your left hand to send a CC/bend value
  4. FL auto-detects and maps it

## Recording in FL Studio

1. Select the channel/instrument you want to record into
2. Click the **record button** (or press R)
3. Enable **Notes & automation** in the recording filter
4. Press play — perform with your hands
5. Stop when done — your pattern now contains the MIDI data

## Controls

- **q** — quit cleanly (sends note_off, closes MIDI port)
- Edit the `CONFIG` block at the top of `hand_instrument.py` to change scale, note range, pinch sensitivity, etc.

## Config Options

```python
MIDI_PORT = "HandMIDI"          # must match your LoopMIDI port name
SCALE = [0, 3, 5, 7, 10]       # pentatonic minor (semitone offsets)
NOTE_MIN = 48                   # C3
NOTE_MAX = 84                   # C6
PINCH_THRESHOLD = 0.05          # lower = harder to trigger
VELOCITY_MIN = 40
VELOCITY_MAX = 127
```

## Original Project

The original [HandTrack_To_MIDI](https://github.com/frkatona/HandTrack_To_MIDI) by frkatona maps single-hand finger flexion to MIDI CC values. This version extends it with two-hand support, note playing via pinch gestures, pitch bend, and a visual HUD overlay.
