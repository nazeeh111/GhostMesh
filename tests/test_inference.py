import json
import numpy as np
import pytest
import torch
from ghostmesh.models import CSIPose
from ghostmesh.replay import infer_recording, predict_frames


def contract():
    torch.set_num_threads(2)
    return CSIPose(width=16), {
        "input_shape": [3, 16, 4],
        "config": {"threads": 2, "data": {"window": 4, "normalize": True}},
    }


def test_annotation_free_inference_and_invalid_suppression(tmp_path, monkeypatch):
    model, checkpoint = contract()
    checkpoint["purpose"] = "software_check_only"
    rng = np.random.default_rng(1)
    csi = rng.uniform(1, 3, (9, 3, 16, 4)).astype("float32")
    csi[5] = np.nan
    recording = tmp_path / "recording.npz"
    np.savez(recording, csi=csi, timestamps=np.arange(9) * 0.2)
    fake = tmp_path / "test.pt"
    fake.write_bytes(b"mocked-checkpoint")
    monkeypatch.setattr(
        "ghostmesh.replay.load_checkpoint", lambda *_: (model, checkpoint)
    )
    output = tmp_path / "replay.json"
    infer_recording(recording, fake, output)
    replay = json.loads(output.read_text())
    assert all("ground_truth" not in f for f in replay["frames"])
    assert replay["frames"][3]["pose"] is not None
    assert replay["frames"][5]["pose"] is None
    assert replay["frames"][6]["pose"] is None
    assert replay["frames"][3]["status"] == "out_of_domain"
    assert replay["provenance"]["kind"] == "synthetic_diagnostic"
    with pytest.raises(FileExistsError):
        infer_recording(recording, fake, output)


def test_sensor_shape_and_timestamp_contracts():
    model, checkpoint = contract()
    x = np.ones((8, 3, 17, 4))
    with pytest.raises(ValueError, match="layouts differ"):
        predict_frames(x, np.arange(8) * 0.1, model, checkpoint)
    x = np.ones((8, 3, 16, 4))
    with pytest.raises(ValueError, match="strictly increase"):
        predict_frames(x, np.zeros(8), model, checkpoint)
