#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 1.9B — Spatial convergence and time-step sensitivity audit

Cooperative Fisher--KPP--chemotaxis system on [0,L] with Neumann
boundary conditions.  This script reuses the even-extension Fourier
pseudospectral IMEX solver defined in ``1.py``.

The audit contains two complementary tests.

1. Spatial convergence (full even-extension FFT grids N=256,512,1024):
   * fully discrete numerical threshold chi_T^num;
   * selected Neumann mode and pattern wavelength;
   * nonlinear branch amplitude and state-profile error.

2. Time-step sensitivity (default dt=1e-3 and 5e-4):
   * branch amplitude at a common physical time;
   * state-profile difference relative to the finest time step.

Required file
-------------
Place this script in the same PyCharm project directory as ``1.py``.

Default behavior
----------------
The default is the paper run, so the green Run button in PyCharm is enough.
The quick preset remains available only for a short diagnostic:
    python task19B_numerical_resolution_audit.py --preset quick --device cuda

Outputs
-------
results_task19B/
    spatial_convergence.csv
    time_step_sensitivity.csv
    numerical_audit_summary.txt
    figure5_numerical_audit.png / pdf
    figure5_profile_overlays.png / pdf
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# Dynamic import of the validated Task 1.6B solver
# ============================================================

HERE = Path(__file__).resolve().parent
TASK16_FILE = HERE / "1.py"
OUT = HERE / "results_task19B"


def load_task16_module():
    if not TASK16_FILE.exists():
        raise FileNotFoundError(
            "Cannot find the base solver '1.py'.\n"
            f"Expected path: {TASK16_FILE}\n"
            "Put this script in the same PyCharm project directory as 1.py."
        )

    module_name = "task16_kpp_chemotaxis"
    spec = importlib.util.spec_from_file_location(module_name, TASK16_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {TASK16_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


task16 = load_task16_module()

ModelParams = task16.ModelParams
SpectralIMEX1D = task16.SpectralIMEX1D
continuous_critical_data = task16.continuous_critical_data
select_mode = task16.select_mode
chi_of_mu = task16.chi_of_mu
chi2_data = task16.chi2_data
initialize_branch_state = task16.initialize_branch_state


# ============================================================
# Run configuration
# ============================================================

@dataclass(frozen=True)
class AuditConfig:
    name: str
    spatial_fft_points: Tuple[int, ...]
    spatial_dt: float
    spatial_t_final: float
    time_fft_points: int
    time_steps: Tuple[float, ...]
    time_t_final: float
    delta_chi: float
    dtype: torch.dtype


def preset_config(name: str) -> AuditConfig:
    name = name.lower()

    if name == "quick":
        return AuditConfig(
            name="quick",
            spatial_fft_points=(256, 512, 1024),
            spatial_dt=1.0e-2,
            spatial_t_final=50.0,
            time_fft_points=1024,
            time_steps=(1.0e-3, 5.0e-4),
            time_t_final=5.0,
            delta_chi=5.0e-3,
            dtype=torch.float64,
        )

    if name == "paper":
        return AuditConfig(
            name="paper",
            spatial_fft_points=(256, 512, 1024),
            spatial_dt=1.0e-2,
            spatial_t_final=500.0,
            time_fft_points=1024,
            time_steps=(1.0e-3, 5.0e-4),
            # Match the T=500 nonlinear branch reported in the manuscript.
            # This is deliberately expensive and should be run on CUDA.
            time_t_final=500.0,
            delta_chi=5.0e-3,
            dtype=torch.float64,
        )

    raise ValueError(f"Unknown preset: {name}")


def validate_config(cfg: AuditConfig) -> None:
    for n_fft in (*cfg.spatial_fft_points, cfg.time_fft_points):
        if n_fft < 32 or n_fft % 2 != 0:
            raise ValueError("Every FFT grid size must be an even integer >= 32.")

    if sorted(cfg.spatial_fft_points) != list(cfg.spatial_fft_points):
        raise ValueError("spatial_fft_points must be strictly increasing.")

    if len(set(cfg.spatial_fft_points)) != len(cfg.spatial_fft_points):
        raise ValueError("spatial_fft_points contains duplicate values.")

    if min(cfg.spatial_dt, cfg.spatial_t_final, cfg.time_t_final) <= 0.0:
        raise ValueError("Time steps and final times must be positive.")

    if any(dt <= 0.0 for dt in cfg.time_steps):
        raise ValueError("All entries in time_steps must be positive.")

    if cfg.delta_chi <= 0.0:
        raise ValueError("delta_chi must be positive.")


# ============================================================
# Journal-style plotting configuration
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 11.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "axes.linewidth": 0.85,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#2166AC"
ORANGE = "#D95F02"
GREEN = "#1B7837"
RED = "#B2182B"
DARK = "#222222"
GREY = "#737373"
LIGHT_GREY = "#D9D9D9"


# ============================================================
# Generic utilities
# ============================================================

def choose_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. "
                "Use --device cpu or check the PyTorch CUDA installation."
            )
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, cpu")


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Sequence[str]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_figure(fig: plt.Figure, basename: str) -> None:
    png_path = OUT / f"{basename}.png"
    pdf_path = OUT / f"{basename}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print("Saved:", png_path)
    print("Saved:", pdf_path)


def relative_l2(state: Tuple[np.ndarray, ...], reference: Tuple[np.ndarray, ...],
                equilibria: Tuple[float, ...]) -> float:
    difference = np.concatenate(
        [(field - ref).ravel() for field, ref in zip(state, reference)]
    )
    reference_perturbation = np.concatenate(
        [(ref - eq).ravel() for ref, eq in zip(reference, equilibria)]
    )
    denominator = np.linalg.norm(reference_perturbation)
    if denominator == 0.0:
        return float(np.linalg.norm(difference))
    return float(np.linalg.norm(difference) / denominator)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13, 1.04, label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def latex_number(value: float) -> str:
    """Compact math-text label, with powers of ten written conventionally."""
    if value <= 0.0:
        return f"{value:g}"
    exponent = int(round(math.log10(value)))
    if math.isclose(value, 10.0 ** exponent, rel_tol=1.0e-12, abs_tol=0.0):
        return rf"10^{{{exponent}}}"
    return f"{value:g}"


# ============================================================
# Fully discrete linear threshold
# ============================================================

def imex_amplification_matrix(
    p: ModelParams,
    mu: float,
    chi: float,
    dt: float,
) -> np.ndarray:
    """Linearization of the actual first-order IMEX splitting."""

    linear_implicit = np.array(
        [
            [-p.d1 * mu + p.r1 - p.k12, p.k12, 0.0],
            [p.k21, -p.d2 * mu + p.r2 - p.k21, 0.0],
            [p.beta / p.tau, 0.0, -(p.Dw * mu + p.alpha) / p.tau],
        ],
        dtype=np.float64,
    )

    linear_explicit = np.array(
        [
            [-2.0 * p.r1, 0.0, chi * mu],
            [0.0, -2.0 * p.r2, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    identity = np.eye(3, dtype=np.float64)
    return np.linalg.solve(
        identity - dt * linear_implicit,
        identity + dt * linear_explicit,
    )


def dominant_discrete_eigenvalue(
    p: ModelParams,
    mu: float,
    chi: float,
    dt: float,
) -> float:
    values = np.linalg.eigvals(imex_amplification_matrix(p, mu, chi, dt))
    return float(np.max(values.real))


def mode_threshold_bisection(
    p: ModelParams,
    mu: float,
    dt: float,
    tolerance: float = 2.0e-13,
) -> float:
    """Solve max Re eig(G_dt)-1=0 without requiring SciPy."""

    def objective(chi: float) -> float:
        return dominant_discrete_eigenvalue(p, mu, chi, dt) - 1.0

    lower = 0.0
    upper = max(2.0, 1.25 * float(chi_of_mu(mu, p)))
    f_lower = objective(lower)
    f_upper = objective(upper)

    if f_lower >= 0.0:
        raise RuntimeError("Unexpected instability at chi=0.")

    expansions = 0
    while f_upper <= 0.0 and expansions < 30:
        upper *= 2.0
        f_upper = objective(upper)
        expansions += 1

    if f_upper <= 0.0:
        raise RuntimeError(f"Could not bracket the threshold for mu={mu:g}.")

    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        f_mid = objective(midpoint)
        if f_mid > 0.0:
            upper = midpoint
        else:
            lower = midpoint
        if upper - lower <= tolerance * max(1.0, abs(midpoint)):
            break

    return 0.5 * (lower + upper)


def numerical_threshold_for_grid(
    p: ModelParams,
    L: float,
    n_fft: int,
    dt: float,
    k_c: float,
) -> Dict[str, float | int]:
    # Linear modes must lie inside the 2/3 retained band.  Only modes near
    # the continuous minimizer can select onset, so a modest extra margin
    # avoids scanning irrelevant highly damped modes.
    retained_max = n_fft // 3
    expected_mode = L * k_c / math.pi
    scan_max = min(retained_max, max(16, int(math.ceil(3.0 * expected_mode))))

    candidates: List[Tuple[float, int, float]] = []
    for n in range(1, scan_max + 1):
        mu_n = (n * math.pi / L) ** 2
        threshold = mode_threshold_bisection(p, mu_n, dt)
        candidates.append((threshold, n, mu_n))

    chi_num, n_num, mu_num = min(candidates, key=lambda item: item[0])
    return {
        "chi_T_num": float(chi_num),
        "n_star_num": int(n_num),
        "mu_star_num": float(mu_num),
        "wavelength_num": float(2.0 * L / n_num),
        "scanned_modes": int(scan_max),
    }


# ============================================================
# Nonlinear branch calculation
# ============================================================

def run_branch_state(
    p: ModelParams,
    L: float,
    n_fft: int,
    dt: float,
    t_final: float,
    delta_chi: float,
    device: torch.device,
    dtype: torch.dtype,
    mode: Dict[str, float | int],
    c2: Dict[str, float],
    tag: str,
) -> Dict[str, object]:
    n_half = n_fft // 2 + 1
    solver = SpectralIMEX1D(p, L, n_half, dt, device, dtype)
    chi = float(mode["chi_T"]) + delta_chi

    u, v, w, amplitude_prediction = initialize_branch_state(
        solver,
        p,
        int(mode["n_star"]),
        float(c2["chi2"]),
        delta_chi,
        c2,
    )

    number_of_steps = int(round(t_final / dt))
    actual_t_final = number_of_steps * dt
    if number_of_steps < 1:
        raise ValueError("The requested final time gives fewer than one step.")

    print(
        f"  {tag}: N={n_fft}, dt={dt:.3e}, "
        f"steps={number_of_steps:,}, T={actual_t_final:g}"
    )

    start = time.perf_counter()
    progress_every = max(1, number_of_steps // 10)
    with torch.no_grad():
        for step in range(number_of_steps):
            u, v, w = solver.step(u, v, w, chi)
            if (step + 1) % progress_every == 0 or step + 1 == number_of_steps:
                print(f"    progress: {100.0 * (step + 1) / number_of_steps:5.1f}%")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    n_star = int(mode["n_star"])
    amplitude = abs(solver.cosine_amplitude(u, n_star, mean=1.0))
    residual = solver.steady_residual(u, v, w, chi)

    spectrum = torch.abs(torch.fft.fft(u - torch.mean(u)))
    search_stop = min(n_fft // 3 + 1, n_fft // 2)
    dominant_n = int(torch.argmax(spectrum[1:search_stop]).detach().cpu()) + 1
    wavelength = 2.0 * L / dominant_n

    stop = n_half
    return {
        "n_fft": n_fft,
        "n_half": n_half,
        "dx": 2.0 * L / n_fft,
        "dt": dt,
        "t_final": actual_t_final,
        "delta_chi": delta_chi,
        "chi": chi,
        "A_prediction": float(amplitude_prediction),
        "A_numeric": float(amplitude),
        "residual": float(residual),
        "dominant_n": dominant_n,
        "wavelength": float(wavelength),
        "elapsed_seconds": float(elapsed),
        "x": solver.x[:stop].detach().cpu().numpy(),
        "u": u[:stop].detach().cpu().numpy(),
        "v": v[:stop].detach().cpu().numpy(),
        "w": w[:stop].detach().cpu().numpy(),
    }


# ============================================================
# Spatial and temporal audits
# ============================================================

def run_spatial_audit(
    cfg: AuditConfig,
    p: ModelParams,
    L: float,
    device: torch.device,
    critical: Dict[str, float],
    mode: Dict[str, float | int],
    c2: Dict[str, float],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    print("\n[1/2] Spatial convergence audit")
    raw_runs: List[Dict[str, object]] = []

    for n_fft in cfg.spatial_fft_points:
        threshold = numerical_threshold_for_grid(
            p, L, n_fft, cfg.spatial_dt, float(critical["k_c"])
        )
        run = run_branch_state(
            p, L, n_fft, cfg.spatial_dt, cfg.spatial_t_final,
            cfg.delta_chi, device, cfg.dtype, mode, c2,
            tag="spatial",
        )
        run.update(threshold)
        raw_runs.append(run)

    reference = raw_runs[-1]
    reference_state = (reference["u"], reference["v"], reference["w"])
    equilibria = (1.0, 1.0, p.beta / p.alpha)
    rows: List[Dict[str, object]] = []

    for run in raw_runs:
        ratio = int(reference["n_fft"]) // int(run["n_fft"])
        if ratio * int(run["n_fft"]) != int(reference["n_fft"]):
            raise ValueError("Spatial grids must be nested for profile comparison.")

        sampled_reference = tuple(field[::ratio] for field in reference_state)
        state = (run["u"], run["v"], run["w"])
        profile_error = relative_l2(state, sampled_reference, equilibria)
        amplitude_error = abs(float(run["A_numeric"]) - float(reference["A_numeric"]))
        amplitude_relative_error = amplitude_error / abs(float(reference["A_numeric"]))

        rows.append({
            "N_fft": run["n_fft"],
            "n_half": run["n_half"],
            "dx": run["dx"],
            "dt": run["dt"],
            "t_final": run["t_final"],
            "chi_T_exact": mode["chi_T"],
            "chi_T_num": run["chi_T_num"],
            "threshold_abs_error": abs(float(run["chi_T_num"]) - float(mode["chi_T"])),
            "n_star_num": run["n_star_num"],
            "wavelength_num": run["wavelength_num"],
            "dominant_n_observed": run["dominant_n"],
            "wavelength_observed": run["wavelength"],
            "A_numeric": run["A_numeric"],
            "A_abs_error_vs_Nmax": amplitude_error,
            "A_rel_error_vs_Nmax": amplitude_relative_error,
            "state_rel_L2_vs_Nmax": profile_error,
            "steady_residual": run["residual"],
            "elapsed_seconds": run["elapsed_seconds"],
        })

    return rows, raw_runs


def run_time_audit(
    cfg: AuditConfig,
    p: ModelParams,
    L: float,
    device: torch.device,
    mode: Dict[str, float | int],
    c2: Dict[str, float],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    print("\n[2/2] Time-step sensitivity audit")
    raw_runs: List[Dict[str, object]] = []

    # Run coarser steps first and use the smallest dt as the reference.
    for dt in sorted(cfg.time_steps, reverse=True):
        raw_runs.append(
            run_branch_state(
                p, L, cfg.time_fft_points, dt, cfg.time_t_final,
                cfg.delta_chi, device, cfg.dtype, mode, c2,
                tag="time",
            )
        )

    reference = min(raw_runs, key=lambda run: float(run["dt"]))
    reference_state = (reference["u"], reference["v"], reference["w"])
    equilibria = (1.0, 1.0, p.beta / p.alpha)
    rows: List[Dict[str, object]] = []

    for run in raw_runs:
        state = (run["u"], run["v"], run["w"])
        profile_error = relative_l2(state, reference_state, equilibria)
        amplitude_error = abs(float(run["A_numeric"]) - float(reference["A_numeric"]))
        amplitude_relative_error = amplitude_error / abs(float(reference["A_numeric"]))

        rows.append({
            "dt": run["dt"],
            "N_fft": run["n_fft"],
            "t_final": run["t_final"],
            "delta_chi": run["delta_chi"],
            "A_numeric": run["A_numeric"],
            "A_abs_diff_vs_finest_dt": amplitude_error,
            "A_rel_diff_vs_finest_dt": amplitude_relative_error,
            "state_rel_L2_vs_finest_dt": profile_error,
            "dominant_n": run["dominant_n"],
            "wavelength": run["wavelength"],
            "steady_residual": run["residual"],
            "elapsed_seconds": run["elapsed_seconds"],
        })

    rows.sort(key=lambda row: float(row["dt"]), reverse=True)
    return rows, raw_runs


# ============================================================
# Output tables, figures, and summary
# ============================================================

def make_audit_figure(
    spatial_rows: List[Dict[str, object]],
    time_rows: List[Dict[str, object]],
) -> None:
    n_values = np.array([row["N_fft"] for row in spatial_rows], dtype=float)
    threshold_error = np.array(
        [row["threshold_abs_error"] for row in spatial_rows], dtype=float
    )
    amplitude_error_space = np.array(
        [row["A_rel_error_vs_Nmax"] for row in spatial_rows], dtype=float
    )
    profile_space = np.array([row["state_rel_L2_vs_Nmax"] for row in spatial_rows], dtype=float)

    dts = np.array([row["dt"] for row in time_rows], dtype=float)
    amplitude_error_time = np.array(
        [row["A_rel_diff_vs_finest_dt"] for row in time_rows], dtype=float
    )
    profile_time = np.array([row["state_rel_L2_vs_finest_dt"] for row in time_rows], dtype=float)
    residual_space = np.array([row["steady_residual"] for row in spatial_rows], dtype=float)
    residual_time = np.array([row["steady_residual"] for row in time_rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.3), constrained_layout=True)

    # Threshold errors are shown directly.  Plotting chi_T itself would
    # magnify meaningless changes in its 13th--15th decimal places.
    ax = axes[0, 0]
    ax.semilogy(n_values, threshold_error, "o-", color=BLUE)
    ax.set_xlabel(r"full FFT points $N$")
    ax.set_ylabel(r"$|\chi_T^{\rm num}-\chi_T|$")
    ax.text(
        0.05, 0.10,
        r"$n_*=5,\quad \Lambda_*=4$",
        transform=ax.transAxes,
        fontsize=9.5,
        color=DARK,
    )
    ax.grid(color=LIGHT_GREY, alpha=0.38, lw=0.65, which="both")
    panel_label(ax, "(a)")

    # Omit the zero reference error at N_max instead of replacing it by an
    # artificial floating-point floor.
    ax = axes[0, 1]
    spatial_mask = n_values < np.max(n_values)
    ax.semilogy(
        n_values[spatial_mask],
        amplitude_error_space[spatial_mask],
        "o-",
        color=GREEN,
        label=r"amplitude error",
    )
    ax.semilogy(
        n_values[spatial_mask],
        profile_space[spatial_mask],
        "s--",
        color=BLUE,
        label=r"state $L^2$ error",
    )
    ax.set_xlabel(r"full FFT points $N$")
    ax.set_ylabel(r"relative difference from $N=1024$")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(color=LIGHT_GREY, alpha=0.38, lw=0.65, which="both")
    panel_label(ax, "(b)")

    # With two prescribed time steps, a quantitative bar comparison is more
    # honest than drawing an apparent convergence curve through two points.
    ax = axes[1, 0]
    finest_dt = np.min(dts)
    temporal_mask = dts > finest_dt
    if np.count_nonzero(temporal_mask) == 1:
        values = [
            amplitude_error_time[temporal_mask][0],
            profile_time[temporal_mask][0],
        ]
        bars = ax.bar(
            [0, 1], values,
            color=[ORANGE, RED],
            width=0.58,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["amplitude", "full state"])
        ax.set_yscale("log")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value * 1.18,
                f"{value:.2e}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    else:
        ax.loglog(
            dts[temporal_mask],
            amplitude_error_time[temporal_mask],
            "o-", color=ORANGE, label="amplitude",
        )
        ax.loglog(
            dts[temporal_mask],
            profile_time[temporal_mask],
            "s--", color=RED, label="full state",
        )
        ax.legend(frameon=False, fontsize=8.5)
    ax.set_xlabel(
        rf"$\Delta t={latex_number(float(np.max(dts)))}$ versus "
        rf"$\Delta t={latex_number(float(finest_dt))}$"
    )
    ax.set_ylabel("relative difference")
    ax.grid(color=LIGHT_GREY, alpha=0.38, lw=0.65, which="both")
    panel_label(ax, "(c)")

    # Residuals assess whether the common terminal time is sufficiently close
    # to the stationary branch.  No reference errors are inserted here.
    ax = axes[1, 1]
    residuals = np.concatenate((residual_space, residual_time))
    positions = np.arange(residuals.size)
    labels = [rf"$N={int(value)}$" for value in n_values] + [
        rf"$\Delta t={latex_number(float(value))}$" for value in dts
    ]
    colors = [BLUE] * len(residual_space) + [RED] * len(residual_time)
    ax.bar(positions, residuals, color=colors, width=0.64)
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_xlabel("resolution setting")
    ax.set_ylabel("stationary residual")
    ax.grid(color=LIGHT_GREY, alpha=0.38, lw=0.65, which="both")
    panel_label(ax, "(d)")

    save_figure(fig, "figure5_numerical_audit")
    plt.close(fig)


def make_profile_figure(
    spatial_runs: List[Dict[str, object]],
    time_runs: List[Dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), constrained_layout=True)

    ax = axes[0]
    spatial_styles = ["--", "-.", "-"]
    spatial_colors = [BLUE, ORANGE, GREEN]
    for run, linestyle, color in zip(spatial_runs, spatial_styles, spatial_colors):
        ax.plot(
            run["x"], run["u"],
            lw=1.65, ls=linestyle, color=color,
            label=f"N={run['n_fft']}",
        )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(T,x)$")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(color=LIGHT_GREY, alpha=0.35, lw=0.6)
    panel_label(ax, "(a)")

    ax = axes[1]
    ordered_time_runs = sorted(
        time_runs, key=lambda item: float(item["dt"]), reverse=True
    )
    time_styles = ["--", "-"]
    time_colors = [BLUE, ORANGE]
    for run, linestyle, color in zip(ordered_time_runs, time_styles, time_colors):
        ax.plot(
            run["x"], run["u"],
            lw=1.65, ls=linestyle, color=color,
            label=rf"$\Delta t={latex_number(float(run['dt']))}$",
        )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(T,x)$")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(color=LIGHT_GREY, alpha=0.35, lw=0.6)
    panel_label(ax, "(b)")

    save_figure(fig, "figure5_profile_overlays")
    plt.close(fig)


def write_summary(
    cfg: AuditConfig,
    p: ModelParams,
    critical: Dict[str, float],
    mode: Dict[str, float | int],
    c2: Dict[str, float],
    spatial_rows: List[Dict[str, object]],
    time_rows: List[Dict[str, object]],
    device: torch.device,
) -> None:
    coarsest_space = spatial_rows[0]
    coarsest_time = max(time_rows, key=lambda row: float(row["dt"]))

    lines = [
        "Task 1.9B — Numerical resolution audit",
        "=" * 52,
        f"preset: {cfg.name}",
        f"device: {device}",
        f"dtype: {cfg.dtype}",
        f"L: 10.0",
        f"continuous chi_c: {float(critical['chi_c']):.12f}",
        f"finite-domain chi_T: {float(mode['chi_T']):.12f}",
        f"selected mode n_star: {int(mode['n_star'])}",
        f"pattern wavelength: {2.0 * 10.0 / int(mode['n_star']):.12f}",
        f"chi_2: {float(c2['chi2']):.12f}",
        "",
        "Spatial convergence:",
        (
            f"  coarsest N={int(coarsest_space['N_fft'])}: "
            f"threshold error={float(coarsest_space['threshold_abs_error']):.6e}, "
            f"state difference={float(coarsest_space['state_rel_L2_vs_Nmax']):.6e}"
        ),
        "",
        "Time-step sensitivity:",
        (
            f"  dt={float(coarsest_time['dt']):.6e} relative to finest dt: "
            f"amplitude difference={float(coarsest_time['A_rel_diff_vs_finest_dt']):.6e}, "
            f"state difference={float(coarsest_time['state_rel_L2_vs_finest_dt']):.6e}"
        ),
    ]
    (OUT / "numerical_audit_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatial convergence and time-step sensitivity audit."
    )
    # Paper is the default so a normal PyCharm Run executes the manuscript
    # calculation.  Use --preset quick only for a short diagnostic.
    parser.add_argument("--preset", choices=("quick", "paper"), default="paper")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--spatial-t-final", type=float, default=None,
        help="Optional override for the spatial-audit final time.",
    )
    parser.add_argument(
        "--time-t-final", type=float, default=None,
        help="Optional override for the time-step-audit final time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = preset_config(args.preset)
    cfg = AuditConfig(
        name=base.name,
        spatial_fft_points=base.spatial_fft_points,
        spatial_dt=base.spatial_dt,
        spatial_t_final=(
            base.spatial_t_final if args.spatial_t_final is None
            else args.spatial_t_final
        ),
        time_fft_points=base.time_fft_points,
        time_steps=base.time_steps,
        time_t_final=(
            base.time_t_final if args.time_t_final is None
            else args.time_t_final
        ),
        delta_chi=base.delta_chi,
        dtype=base.dtype,
    )
    validate_config(cfg)

    device = choose_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    p = ModelParams()
    L = 10.0
    critical = continuous_critical_data(p)
    mode = select_mode(L, p, critical["mu_c"])
    c2 = chi2_data(
        L, p, int(mode["n_star"]), float(mode["mu_star"]), float(mode["chi_T"])
    )

    print("=" * 72)
    print("Task 1.9B — Spatial convergence and time-step sensitivity")
    print("=" * 72)
    print("Preset:", cfg.name)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    print("dtype:", cfg.dtype)
    print(f"chi_T={float(mode['chi_T']):.12f}, n_star={int(mode['n_star'])}")
    print(f"chi_2={float(c2['chi2']):.12f}")

    spatial_rows, spatial_runs = run_spatial_audit(
        cfg, p, L, device, critical, mode, c2
    )
    time_rows, time_runs = run_time_audit(cfg, p, L, device, mode, c2)

    write_csv(
        OUT / "spatial_convergence.csv",
        spatial_rows,
        [
            "N_fft", "n_half", "dx", "dt", "t_final",
            "chi_T_exact", "chi_T_num", "threshold_abs_error",
            "n_star_num", "wavelength_num",
            "dominant_n_observed", "wavelength_observed", "A_numeric",
            "A_abs_error_vs_Nmax", "A_rel_error_vs_Nmax",
            "state_rel_L2_vs_Nmax", "steady_residual", "elapsed_seconds",
        ],
    )

    write_csv(
        OUT / "time_step_sensitivity.csv",
        time_rows,
        [
            "dt", "N_fft", "t_final", "delta_chi", "A_numeric",
            "A_abs_diff_vs_finest_dt", "A_rel_diff_vs_finest_dt",
            "state_rel_L2_vs_finest_dt", "dominant_n", "wavelength",
            "steady_residual", "elapsed_seconds",
        ],
    )

    make_audit_figure(spatial_rows, time_rows)
    make_profile_figure(spatial_runs, time_runs)
    write_summary(cfg, p, critical, mode, c2, spatial_rows, time_rows, device)

    maximum_residual = max(
        [float(row["steady_residual"]) for row in spatial_rows]
        + [float(row["steady_residual"]) for row in time_rows]
    )

    print("\n" + "=" * 72)
    print("Audit finished successfully.")
    print("Results saved in:", OUT)
    print(f"Maximum stationary residual: {maximum_residual:.6e}")
    if maximum_residual > 1.0e-8:
        print(
            "NOTE: the residual is above 1e-8. The resolution comparison is "
            "valid at the common terminal time, but avoid claiming full "
            "stationarity without reporting this residual."
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
