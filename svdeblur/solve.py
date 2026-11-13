"""Iterative MAP restoration for the spatially varying forward model.

Objective (a single parameter set drives every iteration and every export):

    J(x) = 0.5 ||A x - b||^2 / sigma^2
           + lam * TV_eps(x)
           + 0.5 * mu * ||x - x_ref||^2      (optional damping)

* The data term is whitened by the declared Gaussian noise variance sigma^2,
  so lambda is a trade-off against stated measurement confidence.
* TV_eps(x) = sum sqrt(|grad x|^2 + eps^2) is edge-preserving regularisation
  applied to the *inverse-problem solution*, not a generic sharpening filter.
* mu > 0 pins the solution toward a reference (the blurred input by default).
  With mu = 0 the problem is still regularised by TV, but the solver also
  guards against semi-convergence (residual falling then rising while noise
  is amplified) and reports exactly why it stopped.

The optimiser is FISTA with backtracking line search on the smooth objective;
the gradient uses A and A^T, so the adjoint is exercised on every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

import numpy as np

from .operator import SVForward

STOP_REASONS = {
    "rel_residual": "relative data-residual change fell below tolerance",
    "max_iter": "iteration budget exhausted",
    "noise_floor": "residual reached the declared noise floor",
    "semi_convergence": "residual bottomed out then rose while TV kept falling",
    "divergence": "objective rose persistently; stopped to avoid noise blow-up",
    "nonfinite": "non-finite values appeared; stopped",
}


@dataclass
class SolveConfig:
    iters: int = 300
    lam: float | None = None      # None -> calibrated from sigma
    lam_rel: float = 0.02         # lam = lam_rel * sigma when lam is None
    tv_eps: float = 0.008
    sigma: float = 1.0 / 255.0
    damping_mu: float = 0.0
    reference_weight: bool = True   # start from the blurred input
    tol: float = 1e-4
    noise_floor_factor: float = 0.8
    clip: tuple[float, float] = (0.0, 1.0)
    step_safety: float = 1.0
    max_backtrack: int = 20
    semi_window: int = 8
    max_obj_rises: int = 6
    snapshot_iters: tuple[int, ...] = ()

    def resolved_lam(self) -> float:
        return self.lam_rel * self.sigma if self.lam is None else float(self.lam)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["lam_resolved"] = self.resolved_lam()
        return d


@dataclass
class SolveResult:
    image: np.ndarray
    residuals: list[float] = dc_field(default_factory=list)       # ||r||/||b||
    residual_abs: list[float] = dc_field(default_factory=list)    # ||r||
    objectives: list[float] = dc_field(default_factory=list)
    tv_terms: list[float] = dc_field(default_factory=list)
    step_sizes: list[float] = dc_field(default_factory=list)
    snapshots: dict[int, np.ndarray] = dc_field(default_factory=dict)
    stop_reason: str = ""
    stop_detail: str = ""
    stop_iteration: int = 0
    noise_floor: float = 0.0
    config: dict[str, Any] = dc_field(default_factory=dict)
    operator: dict[str, Any] = dc_field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "operator": self.operator,
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
            "stop_iteration": self.stop_iteration,
            "noise_floor": self.noise_floor,
            "iterations_run": len(self.residuals),
            "final_residual_relative": self.residuals[-1] if self.residuals else None,
            "final_residual_abs": self.residual_abs[-1] if self.residual_abs else None,
            "final_objective": self.objectives[-1] if self.objectives else None,
            "final_tv": self.tv_terms[-1] if self.tv_terms else None,
            "residuals": self.residuals,
            "objectives": self.objectives,
            "tv_terms": self.tv_terms,
            "snapshot_iters": sorted(self.snapshots),
        }


def _grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy = np.zeros_like(x)
    gx = np.zeros_like(x)
    gy[:-1, :] = x[1:, :] - x[:-1, :]
    gx[:, :-1] = x[:, 1:] - x[:, :-1]
    return gy, gx


def _grad_adjoint(py: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Exact transpose of _grad: <grad x, p> == <x, grad_adjoint p>."""
    out = np.zeros_like(py)
    out[:-1, :] += py[:-1, :]
    out[1:, :] -= py[:-1, :]
    out[:, :-1] += px[:, :-1]
    out[:, 1:] -= px[:, :-1]
    return -out


def tv_value(x: np.ndarray, eps: float) -> float:
    gy, gx = _grad(x)
    return float(np.sqrt(gy * gy + gx * gx + eps * eps).sum())


def tv_gradient(x: np.ndarray, eps: float) -> np.ndarray:
    gy, gx = _grad(x)
    mag = np.sqrt(gy * gy + gx * gx + eps * eps)
    return _grad_adjoint(gy / mag, gx / mag)


class TVSolver:
    def __init__(self, forward: SVForward, config: SolveConfig, reference: np.ndarray | None = None):
        self.A = forward
        self.cfg = config
        H, W = forward.image_shape
        self.ref = np.zeros((H, W)) if reference is None else np.asarray(reference, float)
        self.pixels = H * W

    # ---------------------------------------------------------- Lipschitz bound
    def _data_lipschitz(self) -> float:
        rng = np.random.default_rng(7)
        H, W = self.A.image_shape
        v = rng.standard_normal((H, W))
        v /= np.linalg.norm(v)
        lam_max = 0.0
        for _ in range(10):
            w = self.A.apply_adjoint(self.A.apply(v))
            lam_max = float(np.linalg.norm(w))
            n = np.linalg.norm(w)
            if n == 0:
                break
            v = w / n
        return lam_max

    def _initial_step(self) -> float:
        """1 / L for the smooth part, from a measured data-term Lipschitz bound.

        The data term is the unweighted 0.5||Ax-b||^2, so L is the top
        eigenvalue of A^T A (power iteration).  sigma enters the noise floor
        and the default regularisation strength, not the step size.
        """
        lip = self._data_lipschitz()
        lip += self.cfg.damping_mu
        return self.cfg.step_safety / max(lip, 1e-12)

    def _objective(self, x: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
        r = self.A.apply(x) - b
        data = 0.5 * float(np.sum(r * r))
        tv = self.cfg.resolved_lam() * tv_value(x, self.cfg.tv_eps)
        damp = 0.5 * self.cfg.damping_mu * float(np.sum((x - self.ref) ** 2))
        return data + tv + damp, data, tv

    def _gradient(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        g = self.A.apply_adjoint(self.A.apply(x) - b)
        g += self.cfg.resolved_lam() * tv_gradient(x, self.cfg.tv_eps)
        if self.cfg.damping_mu > 0:
            g += self.cfg.damping_mu * (x - self.ref)
        return g

    # ---------------------------------------------------------------- main loop
    def run(self, b: np.ndarray, progress: Callable[[int, dict], None] | None = None) -> SolveResult:
        cfg = self.cfg
        H, W = self.A.image_shape
        b = np.asarray(b, dtype=np.float64)
        if b.shape != (H, W):
            raise ValueError(f"input has shape {b.shape}, expected {(H, W)}")

        res = SolveResult(image=np.zeros_like(b))
        res.config = cfg.as_dict()
        res.operator = self.A.describe()
        floor_abs = cfg.noise_floor_factor * cfg.sigma * float(np.sqrt(self.pixels))
        res.noise_floor = floor_abs
        bnorm = max(float(np.linalg.norm(b)), 1e-12)

        x = np.clip(self.ref if cfg.reference_weight else np.full((H, W), 0.5), *cfg.clip)
        z = x.copy()
        t = 1.0
        step = self._initial_step()
        prev_res_abs = None
        low_res, low_iter, rising = np.inf, 0, 0
        prev_obj, rises = np.inf, 0
        snap = set(int(k) for k in cfg.snapshot_iters)

        def snapshot(i: int, img: np.ndarray):
            if i in snap:
                res.snapshots[i] = np.clip(img, *cfg.clip).copy()

        for it in range(1, cfg.iters + 1):
            fz = self._objective(z, b)[0]
            for bt in range(cfg.max_backtrack + 1):
                gz = self._gradient(z, b)
                x_cand = np.clip(z - step * gz, *cfg.clip)
                f_cand = self._objective(x_cand, b)[0]
                # sufficient-decrease (Armijo) check keeps FISTA stable under
                # the clipped feasible set
                if f_cand <= fz - 1e-4 * step * float(np.sum(gz * (z - x_cand))) or bt == cfg.max_backtrack:
                    break
                step *= 0.5
            x_new = x_cand
            obj, data, tv = self._objective(x_new, b)
            r_abs = float(np.sqrt(2.0 * data))
            res.residual_abs.append(r_abs)
            res.residuals.append(r_abs / bnorm)
            res.objectives.append(float(obj))
            res.tv_terms.append(float(tv))
            res.step_sizes.append(float(step))

            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            z = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
            x = x_new
            snapshot(it, x)
            if progress is not None:
                progress(it, {"iteration": it, "residual": res.residuals[-1],
                              "residual_abs": r_abs, "objective": obj,
                              "tv": tv, "step": step})

            def stop(reason: str, detail: str) -> SolveResult:
                res.image = np.clip(x, *cfg.clip)
                res.stop_reason = reason
                res.stop_detail = detail
                res.stop_iteration = it
                if it not in res.snapshots:
                    res.snapshots[it] = res.image.copy()
                return res

            if not np.isfinite(x).all():
                return stop("nonfinite", "non-finite pixel value encountered")
            if r_abs <= floor_abs:
                return stop("noise_floor",
                            f"||r||={r_abs:.3e} <= tau*sigma*sqrt(N)={floor_abs:.3e} "
                            f"(tau={cfg.noise_floor_factor:.2f}, sigma={cfg.sigma:.3e}, "
                            f"N={self.pixels}) at iteration {it}; further iterations "
                            f"would fit noise")
            if prev_res_abs is not None and it > 3:
                rel = abs(prev_res_abs - r_abs) / max(prev_res_abs, 1e-15)
                if rel < cfg.tol:
                    return stop("rel_residual",
                                f"relative residual change {rel:.2e} < tol {cfg.tol:.1e} "
                                f"at iteration {it}")
            prev_res_abs = r_abs

            if r_abs < low_res:
                low_res, low_iter, rising = r_abs, it, 0
            elif it > low_iter:
                rising += 1
                # require a genuine, sustained rise above the low-water mark;
                # a 0.5% margin ignores floating-point jitter
                if rising >= cfg.semi_window and r_abs > low_res * (1.0 + 0.005):
                    return stop("semi_convergence",
                                f"residual bottomed at {low_res:.3e} (iteration {low_iter}) "
                                f"then rose to {r_abs:.3e} over {rising} iterations; "
                                f"noise begins to be amplified, stopping before it grows")
            if obj > prev_obj:
                rises += 1
                if rises >= cfg.max_obj_rises:
                    return stop("divergence",
                                f"objective rose for {rises} consecutive iterations; "
                                f"stopped to avoid noise blow-up")
            else:
                rises = 0
            prev_obj = obj

        res.image = np.clip(x, *cfg.clip)
        res.stop_reason = "max_iter"
        res.stop_detail = f"reached {cfg.iters} iterations; inspect the residual curve"
        res.stop_iteration = cfg.iters
        res.snapshots.setdefault(cfg.iters, res.image.copy())
        return res


def residual_stats(b: np.ndarray, x: np.ndarray, forward: SVForward) -> dict[str, float]:
    r = forward.apply(x) - b
    return {
        "l2": float(np.linalg.norm(r)),
        "rms": float(np.sqrt(np.mean(r ** 2))),
        "relative": float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-12)),
        "mean": float(r.mean()),
        "max_abs": float(np.abs(r).max()),
    }
