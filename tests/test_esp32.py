import csv
import json
import numpy as np
import pytest
from ghostmesh.esp32 import convert_csv


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
