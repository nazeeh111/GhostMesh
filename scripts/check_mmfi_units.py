"""Inspect the stored MM-Fi magnitude convention, independently of pose scores."""

import json
from pathlib import Path

import numpy as np


def inspect(amplitude, phase, tolerance=1e-8):
    if amplitude.shape != phase.shape or amplitude.shape[1:] != (3, 114, 10):
        raise ValueError("Expected matching [N,3,114,10] source fields")
    valid = np.isfinite(amplitude) & np.isfinite(phase)
    if not valid.any():
        raise ValueError("No finite source coefficients")
    a, p = amplitude[valid], phase[valid]
    rows = []
    for name, magnitude, angle in [
        ("20log10 magnitude, radians", 10 ** (a / 20), p),
        ("10log10 magnitude, radians", 10 ** (a / 10), p),
        ("stored values as linear magnitude", a, p),
        ("20log10 magnitude, degrees", 10 ** (a / 20), np.deg2rad(p)),
    ]:
        z = magnitude * np.exp(1j * angle)
        error = np.maximum(abs(z.real - z.real.round()), abs(z.imag - z.imag.round()))
        rows.append(
            {
                "hypothesis": name,
                "finite_coefficients": len(error),
                "both_components_near_integer": int((error < tolerance).sum()),
                "max_component_rounding_residual": float(error.max()),
            }
        )
    return {
        "scope": "Signal-unit inference only; not calibrated power or pose performance",
        "tolerance": tolerance,
        "interpretation": "20log10|z| is equivalent to10log10|z|^2; scale/referenceunknown",
        "hypotheses": rows,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    with np.load(root / "examples/rf-units-source.npz", allow_pickle=False) as a:
        result = inspect(a["CSIamp"], a["CSIphase"])
    assert (
        result["hypotheses"][0]["finite_coefficients"]
        == result["hypotheses"][0]["both_components_near_integer"]
    )
    print(json.dumps(result, indent=2))
