"""Reference-only RBF kernel for residual regression.

The caller supplies residual targets, for example Y_ref - f0(X_ref), without
centering them. This module reads no annotations, metadata, or query targets.
It returns a correction; adding the source prior and choosing beta are caller
operations. Software fixtures for this algebra are not accuracy evidence.
"""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import dgemm, dnrm2
from scipy.spatial.distance import cdist

KIND = "reference_residual_rbf_v1"
MAX_RELATIVE_RESIDUAL = 1e-8


def _matrix(value, name, *, rows=None, columns=None, min_rows=0):
    if np.iscomplexobj(value):
        raise ValueError(f"Expected real-valued {name} matrix")
    value = np.array(value, dtype="float64", copy=True)
    if (
        value.ndim != 2
        or value.shape[0] < min_rows
        or value.shape[1] == 0
        or (rows is not None and value.shape[0] != rows)
        or (columns is not None and value.shape[1] != columns)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"Expected finite {name} matrix with compatible dimensions")
    return value


def _positive_scalar(value, name):
    value = np.asarray(value)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _distance(left, right):
    # Direct differences avoid cancellation in ||x||² + ||y||² - 2 x@y.
    distance = cdist(left, right, metric="sqeuclidean")
    if not np.isfinite(distance).all():
        raise ValueError("Nonfinite squared reference/query distance")
    return distance


def _rbf(distance, bandwidth_squared):
    # Underflow represents a zero far-reference correction, as intended.
    with np.errstate(under="ignore", over="ignore"):
        return np.exp((-0.5 * distance) / bandwidth_squared)


@dataclass(frozen=True)
class ReferenceRBFKernel:
    """Copied, read-only feature state fitted before any targets are supplied.

    Standard deviations use ddof=0. bandwidth_squared is the median of all
    unordered, distinct-reference squared distances after normalization.
    floored_columns marks reference columns whose SD is <= scale_floor.
    """

    input_mean: np.ndarray
    input_scale: np.ndarray
    reference_design: np.ndarray
    bandwidth_squared: float
    scale_floor: float
    floored_columns: np.ndarray

    def __post_init__(self):
        design = _matrix(self.reference_design, "reference", min_rows=2)
        mean = np.array(self.input_mean, dtype="float64", copy=True)
        scale = np.array(self.input_scale, dtype="float64", copy=True)
        floored = np.array(self.floored_columns, copy=True)
        floor = _positive_scalar(self.scale_floor, "scale_floor")
        width = _positive_scalar(self.bandwidth_squared, "bandwidth_squared")
        if (
            mean.shape != (design.shape[1],)
            or scale.shape != mean.shape
            or floored.shape != mean.shape
            or floored.dtype != np.dtype(bool)
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale < floor)
            or floored.all()
            or np.any(scale[floored] != floor)
            or np.any(scale[~floored] <= floor)
        ):
            raise ValueError("Invalid reference normalization state")
        for name, value in (
            ("input_mean", mean),
            ("input_scale", scale),
            ("reference_design", design),
            ("floored_columns", floored),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "scale_floor", floor)
        object.__setattr__(self, "bandwidth_squared", width)

    @classmethod
    def from_reference(cls, reference, *, scale_floor=0.001):
        """Fit feature statistics from reference rows only, never query rows."""
        reference = _matrix(reference, "reference", min_rows=2)
        floor = _positive_scalar(scale_floor, "scale_floor")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                mean = reference.mean(axis=0)
                std = reference.std(axis=0, ddof=0)
                floored = std <= floor
                if floored.all():
                    raise ValueError("All reference columns are near constant")
                scale = np.maximum(std, floor)
                design = (reference - mean) / scale
        except FloatingPointError as exc:
            raise ValueError("Nonfinite reference normalization") from exc
        distance = _distance(design, design)
        width = float(np.median(distance[np.triu_indices(len(design), k=1)]))
        _positive_scalar(width, "median bandwidth_squared")
        return cls(mean, scale, design, width, floor, floored)

    def cross_kernel(self, query):
        """Apply the fixed reference statistics independently to each query."""
        query = _matrix(query, "query", columns=self.reference_design.shape[1])
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                design = (query - self.input_mean) / self.input_scale
        except FloatingPointError as exc:
            raise ValueError("Nonfinite query normalization") from exc
        return _rbf(_distance(design, self.reference_design), self.bandwidth_squared)

    def solve(self, reference_targets, alpha):
        """Solve (K + n_ref * alpha * I) dual = targets, without an intercept.

        Return float64 dual coefficients and ||A dual - targets||_F /
        ||targets||_F (zero for exactly zero targets). Fail above 1e-8.
        """
        targets = _matrix(
            reference_targets, "reference targets", rows=len(self.reference_design)
        )
        alpha = _positive_scalar(alpha, "alpha")
        system = _rbf(
            _distance(self.reference_design, self.reference_design),
            self.bandwidth_squared,
        )
        with np.errstate(over="ignore"):
            ridge = len(system) * alpha
        if not np.isfinite(ridge):
            raise ValueError("Nonfinite n_ref * alpha")
        system.flat[:: len(system) + 1] += ridge
        if not np.isfinite(system).all():
            raise ValueError("Nonfinite regularized kernel")
        dual = cho_solve(cho_factor(system, lower=True), targets)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = dgemm(1.0, system, dual) - targets
        if not np.isfinite(dual).all() or not np.isfinite(residual).all():
            raise ValueError("Nonfinite kernel solution")
        # Common scaling avoids overflow/underflow in the norm ratio itself.
        target_scale = float(np.max(np.abs(targets)))
        if target_scale:
            with np.errstate(over="ignore", invalid="ignore"):
                relative = float(
                    dnrm2((residual / target_scale).ravel())
                    / dnrm2((targets / target_scale).ravel())
                )
        else:
            relative = float(dnrm2(residual.ravel()))
        if not np.isfinite(relative) or relative > MAX_RELATIVE_RESIDUAL:
            raise ValueError("Kernel normal-equation residual exceeds 1e-8")
        return np.asarray(dual, dtype="float64"), relative

    def predict(self, query, dual):
        """Return the correction only; no fitted target mean or source prior."""
        dual = _matrix(dual, "dual", rows=len(self.reference_design))
        cross = self.cross_kernel(query)
        if not len(cross):
            return np.empty((0, dual.shape[1]), dtype="float64")
        correction = dgemm(1.0, cross, dual)
        if not np.isfinite(correction).all():
            raise ValueError("Nonfinite kernel correction")
        return correction

    def state_dict(self):
        """Return independent NPZ-safe arrays, without targets or coefficients."""
        return dict(
            kind=np.asarray(KIND),
            input_mean=self.input_mean.copy(),
            input_scale=self.input_scale.copy(),
            reference_design=self.reference_design.copy(),
            bandwidth_squared=np.asarray(self.bandwidth_squared),
            scale_floor=np.asarray(self.scale_floor),
            floored_columns=self.floored_columns.copy(),
        )

    @classmethod
    def from_state(cls, state: Mapping):
        """Restore feature state from an allow_pickle=False NPZ or mapping."""
        try:
            if np.asarray(state["kind"]).shape != () or str(state["kind"]) != KIND:
                raise ValueError("Unexpected reference kernel kind")
            return cls(
                **{
                    name: state[name]
                    for name in (
                        "input_mean",
                        "input_scale",
                        "reference_design",
                        "bandwidth_squared",
                        "scale_floor",
                        "floored_columns",
                    )
                }
            )
        except KeyError as exc:
            raise ValueError(f"Missing reference kernel state: {exc.args[0]}") from exc
