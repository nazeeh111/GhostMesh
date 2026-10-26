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
uv run ghostmesh esp-record --port /dev/cu.YOUR_DEVICE \
  --out data/session-001 --seconds 60 --baud 921600 \
  --consent-note "Authorized test area and consenting participants"
```

The raw `serial.log` preserves startup lines and malformed rows, and
`metadata.json` records the chosen port, rate, duration and consent note.
Existing output directories are refused. No camera, Bluetooth, network
discovery or pose inference is started by this command.

Start the recorder before resetting the receiver so its CSV header is captured.
A recording started after that header was emitted cannot be decoded by guessing
column offsets. Keep the original log unchanged.

## Convert CSI rows

Pass the recorded `serial.log` directly to the converter. It finds the CSV
header, skips startup/log lines and parses `CSI_DATA` rows. Required header
fields are `local_timestamp` and `data`; the standard firmware also includes
`type`. Both `first_word` and `first_word_invalid` identify the invalid-first-word
flag. Repeated headers are recognized without inventing a new receiver clock.
For another format, call `convert_csv` with explicit timestamp/data field
names or implement a separately tested adapter rather than guessing offsets.

```bash
uv run ghostmesh esp-convert --source data/session-001/serial.log \
  --out data/session-001/amplitude.npz
```

Signed-eight-bit integer I/Q pairs become magnitudes. The receiver's microsecond timestamp
counter is unwrapped; malformed/out-of-order rows are rejected and counted.
Mixed subcarrier lengths are refused. Invalid first-word subcarriers become
missing values. The NPZ contains signal and timestamps, with no invented pose
labels. The sidecar counts ignored log lines, recognized headers, rejected rows
and unsupported payloads. Existing NPZ or JSON outputs are refused.

The [official receiver source](https://github.com/espressif/esp-csi/blob/master/examples/get-started/csi_recv/main/app_main.c)
prints `first_word` from `info->first_word_invalid`. Some current firmware paths
also emit gain-compensated or packed wider values. Payloads outside signed-eight-bit
integers are explicitly unsupported here: they are rejected, never clipped or
silently reinterpreted. A compatible raw export or a separately validated
firmware-specific adapter is required. Receiver resets do not synchronize clocks.

Tests use recorded-log-shaped fixtures for both official header layouts,
startup/interleaved log lines, first-word masking and clock-wrap handling.
No physical serial device was opened to validate this adapter.

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
