"""Sealed recovery for the source-only geometric distillation experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def finite_tree(value):
    if torch.is_tensor(value):
        if (
            value.is_complex()
            or value.layout != torch.strided
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Recovery contains a nonfinite/complex/non-strided tensor")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Recovery contains a nonfinite scalar")
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("Unknown recovery value type")


def save(root, model, optimizer, step, identity, arm, device):
    root = Path(root)
    if type(step) is not int or step < 0 or not re.fullmatch(r"[a-z_]+", arm):
        raise ValueError("Invalid recovery step/arm")
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{step:04}.pt"
    payload = dict(
        step=step,
        identity=identity,
        arm=arm,
        device=device,
        model={key: value.detach().cpu() for key, value in model.state_dict().items()},
        optimizer=optimizer.state_dict(),
        rng_cpu=torch.get_rng_state(),
        rng_mps=torch.mps.get_rng_state() if device == "mps" else None,
    )
    finite_tree(payload)
    writing = root / (name + ".writing")
    with writing.open("xb") as stream:
        torch.save(payload, stream)
    # Exclusive final link prevents a second writer from replacing a checkpoint.
    os.link(writing, root / name)
    writing.unlink()
    seal = dict(
        kind="source-geometric-recovery",
        step=step,
        identity=identity,
        arm=arm,
        device=device,
        checkpoint_sha256=digest(root / name),
    )
    with (root / (name + ".json")).open("x") as stream:
        json.dump(seal, stream, indent=2)
        stream.write("\n")


def validate_optimizer(saved, optimizer, step):
    expected = optimizer.state_dict()
    if set(saved) != {"param_groups", "state"} or len(saved["param_groups"]) != len(
        expected["param_groups"]
    ):
        raise ValueError("Recovery optimizer groups differ")
    parameters = {}
    for actual, template, live in zip(
        saved["param_groups"], expected["param_groups"], optimizer.param_groups
    ):
        if actual != template:
            raise ValueError(
                "Recovery optimizer hyperparameters/parameter order differ"
            )
        parameters.update(zip(template["params"], live["params"]))
    if not set(saved["state"]) <= set(parameters) or (step > 0 and not saved["state"]):
        raise ValueError("Recovery optimizer parameter states are incomplete/unknown")
    for index, state in saved["state"].items():
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Unexpected AdamW recovery fields")
        count = state["step"]
        if (
            not torch.is_tensor(count)
            or count.shape != ()
            or count.is_complex()
            or not count.is_floating_point()
            or float(count) != step
        ):
            raise ValueError("Optimizer step differs from checkpoint step")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor, parameter = state[key], parameters[index]
            if (
                not torch.is_tensor(tensor)
                or tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
            ):
                raise ValueError("Recovery moment shape/dtype differs")
        if bool((state["exp_avg_sq"] < 0).any()):
            raise ValueError("Recovery second moments must be nonnegative")


def load(root, model, optimizer, identity, arm, device):
    root = Path(root)
    checkpoints = sorted(root.glob("step-*.pt"))
    if not checkpoints:
        if root.is_dir() and any(root.iterdir()):
            raise ValueError("Recovery artifacts exist without a sealed checkpoint")
        return 0
    if any(
        not path.with_name(path.name + ".json").is_file() for path in checkpoints
    ) or any(root.glob("*.writing")):
        raise ValueError(
            "Unsealed partial checkpoint requires explicit recovery review"
        )
    path = checkpoints[-1]
    match = re.fullmatch(r"step-(\d{4})\.pt", path.name)
    if not match:
        raise ValueError("Recovery checkpoint filename differs")
    step = int(match[1])
    seal = json.loads(path.with_name(path.name + ".json").read_text())
    if type(seal.get("step")) is not int:
        raise ValueError("Recovery seal step must be an integer")
    if seal != dict(
        kind="source-geometric-recovery",
        step=step,
        identity=identity,
        arm=arm,
        device=device,
        checkpoint_sha256=digest(path),
    ):
        raise ValueError("Recovery seal identity/step/arm/device/hash differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or type(saved.get("step")) is not int:
        raise ValueError("Recovery payload step must be an integer")
    if set(saved) != {
        "step",
        "identity",
        "arm",
        "device",
        "model",
        "optimizer",
        "rng_cpu",
        "rng_mps",
    } or any(
        saved[key] != expected
        for key, expected in (
            ("step", step),
            ("identity", identity),
            ("arm", arm),
            ("device", device),
        )
    ):
        raise ValueError("Recovery payload differs from its filename/seal")
    finite_tree(saved)
    expected = model.state_dict()
    if set(saved["model"]) != set(expected):
        raise ValueError("Recovery model tensor set differs")
    for key, value in saved["model"].items():
        if (
            not torch.is_tensor(value)
            or value.shape != expected[key].shape
            or value.dtype != expected[key].dtype
        ):
            raise ValueError("Recovery model tensor shape/dtype differs")
    validate_optimizer(saved["optimizer"], optimizer, step)
    for key in ["rng_cpu", "rng_mps"] if device == "mps" else ["rng_cpu"]:
        value = saved[key]
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.uint8
            or value.ndim != 1
            or not len(value)
        ):
            raise ValueError("Invalid recovery RNG state")
        expected_rng = (
            torch.get_rng_state() if key == "rng_cpu" else torch.mps.get_rng_state()
        )
        if value.shape != expected_rng.shape:
            raise ValueError("Recovery RNG length differs")
    # Preflight CPU generator parsing without changing the global generator.
    torch.Generator().set_state(saved["rng_cpu"])
    if device != "mps" and saved["rng_mps"] is not None:
        raise ValueError("CPU recovery unexpectedly contains an MPS RNG")
    # All identity, schema and moment checks precede mutation of live state.
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng_cpu"])
    if device == "mps":
        torch.mps.set_rng_state(saved["rng_mps"])
    return step
