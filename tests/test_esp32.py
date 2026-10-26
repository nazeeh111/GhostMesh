import csv
import json
import numpy as np
import pytest
from ghostmesh.esp32 import convert_csv


OFFICIAL_HEADERS = [
    "type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_format,len,first_word,data",
    "type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,data",
]


def official_row(fields, timestamp, payload, first_word=0):
    values = dict.fromkeys(fields, "0")
    values.update(
        type="CSI_DATA",
        mac="00:00:00:00:00:00",
        rssi="-30",
        local_timestamp=str(timestamp),
        first_word=str(first_word),
        len=str(len(payload)),
        data=json.dumps(payload),
    )
    return [values[field] for field in fields]


def test_csv_amplitude_rollover_and_invalid_rows(tmp_path):
    p = tmp_path / "signal.csv"
    with p.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "local_timestamp", "data"])
        writer.writerow(["CSI_DATA", 2**32 - 10, json.dumps([3, 4, 5, 12])])
        writer.writerow(["CSI_DATA", 10, json.dumps([0, 5, 8, 15])])
        writer.writerow(["CSI_DATA", 11, "[3]"])
    out = tmp_path / "record.npz"
    report = convert_csv(p, out)
    x = np.load(out)
    np.testing.assert_allclose(x["csi"][:, 0, :, 0], [[5, 13], [5, 17]])
    np.testing.assert_allclose(x["timestamps"], [0, 0.00002], atol=1e-9)
    assert report["rejected_rows"] == 1
    assert "pose" not in x
    with pytest.raises(FileExistsError):
        convert_csv(p, out)


def test_rejected_rollover_row_does_not_advance_receiver_clock(tmp_path):
    p = tmp_path / "signal.csv"
    with p.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "local_timestamp", "data"])
        writer.writerow(["CSI_DATA", 2**32 - 10, "[3,4,5,12]"])
        writer.writerow(["CSI_DATA", 5, "[3,4,5,12,8,15]"])
        writer.writerow(["CSI_DATA", 10, "[3,4,5,12]"])
    out = tmp_path / "record.npz"
    report = convert_csv(p, out)
    np.testing.assert_allclose(np.load(out)["timestamps"], [0, 0.00002], atol=1e-9)
    assert report["rejected_rows"] == 1


@pytest.mark.parametrize("header", OFFICIAL_HEADERS)
def test_unedited_serial_log_detects_official_headers_and_first_word_alias(
    tmp_path, header
):
    source = tmp_path / "serial.log"
    fields = header.split(",")
    payload = [3, 4, 5, 12, 8, 15, 7, 24]
    with source.open("w", newline="") as stream:
        stream.write("ESP-ROM: test-owned boot fixture\n")
        stream.write("I (100) csi_recv: receiver initialized\n")
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerow(official_row(fields, 100, payload, first_word=1))
        stream.write("I (200) csi_recv: compensate_gain 2.0, agc_gain 4\n")
        writer.writerow(official_row(fields, 200, [256, 0, 3, 4, 8, 15, 7, 24]))
        writer.writerow(fields)  # Repeated header is recognized, never decoded as data.
        writer.writerow(official_row(fields, 300, payload))
        stream.write("\x00W (400) csi_recv: test-owned trailing log\n")
    output = tmp_path / "amplitude.npz"
    report = convert_csv(source, output)
    with np.load(output) as arrays:
        assert arrays["csi"].shape == (2, 1, 4, 1)
        assert np.isnan(arrays["csi"][0, 0, :2, 0]).all()
        np.testing.assert_allclose(arrays["csi"][0, 0, 2:, 0], [17, 25])
        np.testing.assert_allclose(arrays["csi"][1, 0, :, 0], [5, 13, 17, 25])
        np.testing.assert_allclose(arrays["timestamps"], [0, 0.0002])
        assert "pose" not in arrays
    assert report["frames"] == 2
    assert report["headers_seen"] == 2
    assert report["ignored_log_lines"] == 4
    assert report["rejected_rows"] == 1
    assert report["unsupported_payload_rows"] == 1
    assert json.loads(output.with_suffix(".json").read_text()) == report


@pytest.mark.parametrize("payload", [[128, 0, 1, 2], [1.5, 0, 1, 2], [True, 0, 1, 2]])
def test_wider_or_noninteger_payloads_are_rejected_without_clipping(tmp_path, payload):
    source = tmp_path / "signal.csv"
    with source.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["type", "local_timestamp", "data"])
        writer.writerow(["CSI_DATA", 100, json.dumps(payload)])
    output = tmp_path / "amplitude.npz"
    with pytest.raises(
        ValueError, match="unsupported signed-eight-bit payload rows: 1"
    ):
        convert_csv(source, output)
    assert not output.exists()


def test_missing_header_does_not_guess_offsets_or_create_output(tmp_path):
    source = tmp_path / "serial.log"
    source.write_text(
        'I (100): recording began after firmware header\nCSI_DATA,100,"[3,4,5,12]"\n'
    )
    output = tmp_path / "amplitude.npz"
    with pytest.raises(ValueError, match="No CSI CSV header found"):
        convert_csv(source, output)
    assert not output.exists()


def test_partial_log_line_is_ignored_but_partial_csi_row_is_counted(tmp_path):
    source = tmp_path / "serial.log"
    source.write_text(
        'type,local_timestamp,data\nI (100): test log,"unfinished\nCSI_DATA,100,"[3,4,5,12]"\nCSI_DATA,200,"[3,4\n'
    )
    report = convert_csv(source, tmp_path / "amplitude.npz")
    assert report["frames"] == 1
    assert report["ignored_log_lines"] == 1
    assert report["rejected_rows"] == 1


def test_conflicting_first_word_aliases_are_not_silently_chosen(tmp_path):
    source = tmp_path / "signal.csv"
    with source.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["type", "local_timestamp", "first_word", "first_word_invalid", "data"]
        )
        writer.writerow(["CSI_DATA", 100, 1, 0, "[3,4,5,12]"])
        writer.writerow(["CSI_DATA", 200, 0, 0, "[3,4,5,12]"])
    report = convert_csv(source, tmp_path / "amplitude.npz")
    assert report["frames"] == 1
    assert report["rejected_rows"] == 1


def test_existing_sidecar_is_preserved_and_output_suffix_is_explicit(tmp_path):
    output = tmp_path / "amplitude.npz"
    output.with_suffix(".json").write_text("preserve these test-owned bytes")
    with pytest.raises(FileExistsError):
        convert_csv(tmp_path / "missing.csv", output)
    assert output.with_suffix(".json").read_text() == "preserve these test-owned bytes"
    with pytest.raises(ValueError, match=".npz suffix"):
        convert_csv(tmp_path / "missing.csv", tmp_path / "ambiguous-extension")
