import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_kernels.py"
    spec = importlib.util.spec_from_file_location("benchmark_kernels", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_benchmark_verifies_and_reports_only_temporal_scope(tmp_path):
    value = module()
    output = tmp_path / "benchmark.json"
    result = value.run_benchmark(
        output,
        widths=(8,),
        threads=(1,),
        timesteps=4,
        batch_size=2,
        repeats=1,
        warmup=1,
    )
    assert output.is_file()
    assert result["parity"]["status"] == "passed"
    assert len(result["parity"]["cases"]) == 4
    assert result["test_data_used"] is False
    assert "pose accuracy" in result["excludes"]
    assert (
        result["measurements"][0]["statistics_ms"]["direct"]["train_cpu_ms"]["median"]
        > 0
    )
    assert "ghostmesh/models.py" in result["source_sha256"]
    with pytest.raises(FileExistsError):
        value.run_benchmark(output)


def test_benchmark_rejects_invalid_configuration_before_output(tmp_path):
    value = module()
    with pytest.raises(ValueError):
        value.run_benchmark(tmp_path / "invalid.json", widths=(7,))
    assert not (tmp_path / "invalid.json").exists()
