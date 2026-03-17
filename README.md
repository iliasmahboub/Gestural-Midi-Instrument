# Hand Instrument

This is a small webcam instrument for FL Studio and any other MIDI-friendly setup that feels like playing light in the air.

It started from [HandTrack_To_MIDI](https://github.com/frkatona/HandTrack_To_MIDI), then turned into something more melodic: two hands, playable scales, smoother expression, and a little stage presence on screen so you can actually feel what the instrument is doing while you perform.

## What It Does

| Hand | Role |
|------|------|
| **Right hand** | Melody. Wrist height chooses the note, pinch plays it, finger spread shapes velocity. |
| **Left hand** | Expression. Wrist height sends pitch bend, index finger height sends mod wheel. |

The note lane is quantized to a chosen scale, so you can stay musical while still moving freely.

## Musical Stuff Added

- Named scales now live inside the app instead of one hard-coded interval list.
- Roots are changeable in real time, so getting to **E natural minor**, **A phrygian**, or **D hijaz** is immediate.
- Added darker and more exotic colors like:
  - Major
  - Minor Pentatonic
  - Major Pentatonic
  - Major Bebop
  - Mixolydian
  - Lydian
  - Natural Minor
  - Harmonic Minor
  - Melodic Minor
  - Minor Bebop
  - Dorian
  - Phrygian
  - Phrygian Dominant
  - Double Harmonic
  - Hijaz
  - Bayati Approx
  - Kurd
  - Sika Approx
  - Nahawand
  - Whole Tone
- Pinch triggering now uses hysteresis, which makes it feel less twitchy and more playable.
- Pitch bend and mod wheel are smoothed a bit, so expression feels more sung than jittered.
- The HUD now shows the current scale, root, note map, and a scale picker overlay grouped by musical family.

Note on the Arabic entries: these are **12-tone equal temperament approximations**, not strict maqam intonation. They are useful, expressive, and fun, but they are still living inside standard MIDI note spacing.

## Controls

- `q` quits cleanly
- `s` opens or closes the scale picker
- `[` and `]` cycle scales
- `,` and `.` cycle roots
- Arrow keys also work inside the scale picker

## Setup

### 1. Install LoopMIDI

- Download it from [Tobias Erichsen's site](https://www.tobias-erichsen.de/software/loopmidi.html)
- Create a virtual port named `HandMIDI`
- Leave it running

### 2. Configure FL Studio

- Open `Options > MIDI Settings`
- Find `HandMIDI` under input
- Enable it
- Set the port number you want

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Use Python 3.10 to 3.12. MediaPipe does not yet support 3.13 here.

### 4. Run the instrument

```bash
python hand_instrument.py
```

## Recording in FL Studio

1. Select the instrument/channel you want to record.
2. Hit record.
3. Make sure notes and automation are enabled.
4. Perform with your hands.
5. Stop and edit the captured MIDI like any other part.

## Mapping Expression

Pitch bend goes out as pitch wheel data. Modulation goes out as `CC1`.

To map hand movement to a parameter in FL Studio:

1. Right-click a knob or slider.
2. Choose `Link to controller...`
3. Move your left hand.
4. Let FL learn the incoming MIDI.

## Config

Most of the personality lives near the top of [`hand_instrument.py`](/C:/Users/ilyas/hand-midi-instrument/hand_instrument.py):

- `NOTE_MIN` and `NOTE_MAX` change the range
- `VELOCITY_MIN` and `VELOCITY_MAX` shape the dynamic floor and ceiling
- `PINCH_ON_THRESHOLD` and `PINCH_OFF_THRESHOLD` change note trigger feel
- `BEND_SMOOTHING` and `MOD_SMOOTHING` change expression glide
- `SCALE_LIBRARY` is where the available modes live

## Why I Like It

It is not trying to be a piano. It is closer to a camera-fed ribbon instrument with a pinch gate and a strange memory for scales. Sometimes it behaves like a flute, sometimes like a synth lead, sometimes like you are conducting a machine that only understands gesture and tension.

That is the point.
