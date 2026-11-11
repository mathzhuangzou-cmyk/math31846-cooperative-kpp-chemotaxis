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

Recommended commands
--------------------
Fast check:
    python task19B_numerical_resolution_audit.py --preset quick --device cuda

Paper run (RTX 4060 or similar GPU):
    python task19B_numerical_resolution_audit.py --preset paper --device cuda

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
    # Endpoint-weighted norm on the common [0,L] grid.
    weights = np.ones(len(reference[0])); weights[[0,-1]] = 0.5
    numerator = sum(np.dot(weights, (field-ref)**2) for field,ref in zip(state,reference))
    denominator = sum(np.dot(weights, (ref-eq)**2) for ref,eq in zip(reference,equilibria))
    return float(np.sqrt(numerator/denominator)) if denominator > 0 else float(np.sqrt(numerator))


def prolong_even_half(field, target_M):
    """Evaluate the same even trigonometric interpolant on the fine grid."""
    full=np.concatenate([field,field[-2:0:-1]])
    old_M=len(full); coeff=np.fft.rfft(full)/old_M
    padded=np.zeros(target_M//2+1,dtype=complex)
    padded[:len(coeff)]=coeff
    if target_M>old_M: padded[old_M//2]*=0.5
    return np.fft.irfft(padded*target_M,n=target_M)[:target_M//2+1]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13, 1.04, label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


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
    return float(np.max(np.abs(values)))


def mode_threshold_bisection(
    p: ModelParams,
    mu: float,
    dt: float,
    tolerance: float = 2.0e-13,
) -> float:
    """Solve spectral_radius(G_dt)-1=0 without requiring SciPy."""

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
    retained_max = (n_fft - 1) // 3
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

        state = tuple(prolong_even_half(run[key], int(reference["n_fft"])) for key in ["u","v","w"])
        profile_error = relative_l2(state, reference_state, equilibria)
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
    chi_num = np.array([row["chi_T_num"] for row in spatial_rows], dtype=float)
    chi_exact = float(spatial_rows[0]["chi_T_exact"])
    amplitudes_space = np.array([row["A_numeric"] for row in spatial_rows], dtype=float)
    profile_space = np.array([row["state_rel_L2_vs_Nmax"] for row in spatial_rows], dtype=float)

    dts = np.array([row["dt"] for row in time_rows], dtype=float)
    amplitudes_time = np.array([row["A_numeric"] for row in time_rows], dtype=float)
    profile_time = np.array([row["state_rel_L2_vs_finest_dt"] for row in time_rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.3), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(n_values, chi_num, "o-", color=BLUE, label=r"$\chi_T^{\rm num}$")
    ax.axhline(chi_exact, color=DARK, ls="--", lw=1.1, label=r"$\chi_T$")
    ax.set_xlabel(r"full FFT points $N$")
    ax.set_ylabel(r"threshold $\chi_T$")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=LIGHT_GREY, alpha=0.38, lw=0.65)
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    ax.plot(n_values, amplitudes_space, "o-", color=GREEN)
    ax.set_xlabel(r"full FFT points $N$")
    ax.set_ylabel(r"branch amplitude $A_{\rm num}$")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(color=LIGHT_GREY, alpha=0.38, lw=0.65)
    panel_label(ax, "(b)")

    ax = axes[1, 0]
    order = np.argsort(dts)
    ax.plot(dts[order], amplitudes_time[order], "o-", color=ORANGE)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel(r"branch amplitude $A_{\rm num}$")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(color=LIGHT_GREY, alpha=0.38, lw=0.65)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    floor = 1.0e-16
    combined_errors = np.concatenate((profile_space, profile_time))
    positions = np.arange(combined_errors.size)
    labels = (
        [rf"$N={int(value)}$" for value in n_values]
        + [rf"$\Delta t={value:.0e}$" for value in dts]
    )
    ax.semilogy(
        positions[:profile_space.size],
        np.maximum(profile_space, floor),
        "o-",
        color=BLUE,
        label=r"spatial: vs. $N_{\max}$",
    )
    ax.semilogy(
        positions[profile_space.size:],
        np.maximum(profile_time, floor),
        "s-",
        color=RED,
        label=r"temporal: vs. $\Delta t_{\min}$",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_xlabel("resolution setting")
    ax.set_ylabel("relative state difference")
    ax.legend(frameon=False, fontsize=8.5)
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
    for run in spatial_runs:
        ax.plot(run["x"], run["u"], lw=1.5, label=f"N={run['n_fft']}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(T,x)$")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(color=LIGHT_GREY, alpha=0.35, lw=0.6)
    panel_label(ax, "(a)")

    ax = axes[1]
    for run in sorted(time_runs, key=lambda item: float(item["dt"]), reverse=True):
        ax.plot(
            run["x"], run["u"], lw=1.5,
            label=rf"$\Delta t={float(run['dt']):.0e}$",
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
        f"delta_chi: {cfg.delta_chi}",
        f"spatial_t_final: {cfg.spatial_t_final}",
        f"time_t_final: {cfg.time_t_final}",
        "Profile metric: endpoint-weighted joint perturbation L2 on the common finest grid.",
        "The comparisons concern finite-time states, not certified equilibria.",
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
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
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
            "n_star_num", "wavelength_num", "A_numeric",
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

    print("\n" + "=" * 72)
    print("Audit finished successfully.")
    print("Results saved in:", OUT)
    print("=" * 72)


if __name__ == "__main__":
    main()
