# Separate ESP32 acquisition mode

The dataset application works without hardware. Live ESP32 pose estimation is
not validated in this project: no consenting capture session, matched labeled
ESP32 dataset or device-specific trained checkpoint has been supplied.

Use [Espressif ESP-CSI](https://github.com/espressif/esp-csi), a separately
licensed Apache-2.0 project, for supported firmware and board instructions.
GhostMesh does not bundle or flash firmware. Choose the exact board and
firmware output before recording; do not assume a six-board research setup
can be replaced by one receiver without retraining.

## Record an authorized session

Install `.[hardware]` or use `uv sync --locked --extra hardware`. Select the
actual serial port yourself. The command never searches, opens or pairs
unrequested devices:

```bash
ghostmesh esp-record --port /dev/cu.YOUR_DEVICE \
  --out data/session-001 --seconds 60 --baud 921600 \
  --consent-note "Authorized test area and consenting participants"
```

The raw `serial.log` preserves startup lines and malformed rows, and
`metadata.json` records the chosen port, rate, duration and consent note.
Existing output directories are refused. No camera, Bluetooth, network
discovery or pose inference is started by this command.

## Convert CSI rows

Retain the original log. Extract its CSV header and CSI_DATA rows into
`csi.csv` using the firmware's actual column names. Conversion expects header
fields `type`, `local_timestamp`, `data` and optionally `first_word_invalid`.
For another format, call `convert_csv` with explicit timestamp/data field
names or implement a separately tested adapter rather than guessing offsets.

```bash
ghostmesh esp-convert --source data/session-001/csi.csv \
  --out data/session-001/amplitude.npz
```

Signed I/Q pairs become magnitudes. The receiver's microsecond timestamp
counter is unwrapped; malformed/out-of-order rows are rejected and counted.
Mixed subcarrier lengths are refused. Invalid first-word subcarriers become
missing values. The NPZ contains signal and timestamps, with no invented pose
labels. Tests validate numerical conversion and timestamp wrap handling.

## Requirements before real pose inference

MM-Fi uses a three-antenna CSI layout with 114 subcarriers and ten packets per
released frame. A single ESP32 recording has a different layout and channel
response. GhostMesh enforces full input shape and cannot transfer MM-Fi
weights by reshaping a capture.

A live pose system needs synchronized compatible links, clock calibration,
adequate packet coverage, consenting paired training labels, a declared 2D
coordinate convention, device-specific training and independent people/room
evaluation. Receiver-local timestamps alone do not synchronize several
devices. Preserve raw packets and calibration evidence. The current signal
recorder and CSV adapter are verified software components; a hardware session
and ESP32 pose accuracy remain unverified.
