#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 1.6B — GPU numerical validation for the cooperative Fisher–KPP–chemotaxis model

Model on (0,L) with Neumann boundary conditions:
    u_t = d1 u_xx - chi (u w_x)_x + r1 u(1-u) + k12 (v-u)
    v_t = d2 v_xx + r2 v(1-v) + k21 (u-v)
    tau w_t = Dw w_xx - alpha w + beta u

The code uses an even 2L-periodic extension, FFT pseudospectral derivatives,
2/3 dealiasing for nonlinear products, and a first-order IMEX step:
linear diffusion/reaction/exchange/signal terms implicit, nonlinear logistic
and chemotactic terms explicit.

Main validations:
  1) continuous threshold chi(mu), quartic root mu_c, intrinsic k_c;
  2) finite-domain mode selection n_*(L);
  3) below/above-threshold growth-rate test;
  4) nonlinear stationary-branch slope chi-chi_T ~ chi_2 A^2.

Designed for an RTX 4060-class GPU. CUDA is used automatically when available.

Examples
--------
Quick GPU run:
    python task16_kpp_chemotaxis_gpu.py --preset quick --device cuda

Paper-quality run:
    python task16_kpp_chemotaxis_gpu.py --preset paper --device cuda

CPU smoke test:
    python task16_kpp_chemotaxis_gpu.py --preset quick --device cpu --skip-branch
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# Configuration
# ============================================================

@dataclass
class ModelParams:
    d1: float = 0.2
    d2: float = 0.1
    Dw: float = 1.0
    r1: float = 1.0
    r2: float = 0.8
    k12: float = 0.3
    k21: float = 0.2
    alpha: float = 1.0
    beta: float = 1.0
    tau: float = 1.0


@dataclass
class RunConfig:
    preset: str
    L: float
    n_half: int
    dt_cross: float
    t_cross: float
    dt_branch: float
    t_branch: float
    sample_dt: float
    eps_cross: float
    ratios: Tuple[float, float]
    branch_deltas: Tuple[float, ...]
    branch_tol: float
    branch_check_dt: float
    L_min: float
    L_max: float
    L_points: int
    mu_plot_min: float
    mu_plot_max: float
    mu_plot_points: int


def preset_config(name: str) -> RunConfig:
    name = name.lower()
    if name == "quick":
        return RunConfig(
            preset="quick",
            L=10.0,
            n_half=257,          # full periodic grid M=512
            dt_cross=0.02,
            t_cross=80.0,
            dt_branch=0.03,
            t_branch=250.0,
            sample_dt=0.2,
            eps_cross=1.0e-4,
            ratios=(0.98, 1.02),
            branch_deltas=(0.01, 0.02, 0.04),
            branch_tol=2.0e-6,
            branch_check_dt=1.0,
            L_min=1.0,
            L_max=30.0,
            L_points=160,
            mu_plot_min=0.05,
            mu_plot_max=10.0,
            mu_plot_points=500,
        )
    if name == "paper":
        return RunConfig(
            preset="paper",
            L=10.0,
            n_half=513,          # full periodic grid M=1024
            dt_cross=0.01,
            t_cross=120.0,
            dt_branch=0.02,
            t_branch=500.0,
            sample_dt=0.1,
            eps_cross=5.0e-5,
            ratios=(0.98, 1.02),
            branch_deltas=(0.0025, 0.005, 0.01, 0.02),
            branch_tol=3.0e-7,
            branch_check_dt=1.0,
            L_min=1.0,
            L_max=40.0,
            L_points=280,
            mu_plot_min=0.03,
            mu_plot_max=12.0,
            mu_plot_points=900,
        )
    if name == "stress":
        return RunConfig(
            preset="stress",
            L=10.0,
            n_half=1025,         # full periodic grid M=2048
            dt_cross=0.005,
            t_cross=160.0,
            dt_branch=0.01,
            t_branch=800.0,
            sample_dt=0.05,
            eps_cross=2.5e-5,
            ratios=(0.98, 1.02),
            branch_deltas=(0.00125, 0.0025, 0.005, 0.01, 0.02),
            branch_tol=8.0e-8,
            branch_check_dt=0.5,
            L_min=1.0,
            L_max=50.0,
            L_points=400,
            mu_plot_min=0.02,
            mu_plot_max=15.0,
            mu_plot_points=1400,
        )
    raise ValueError(f"Unknown preset: {name}")


# ============================================================
# Analytic formulas
# ============================================================

def quartic_coefficients(p: ModelParams) -> np.ndarray:
    a = p.r1 + p.k12
    b = p.r2 + p.k21
    K = p.k12 * p.k21
    Dscr = a * b - K
    return np.array(
        [
            p.Dw * p.d1 * p.d2**2,
            2.0 * p.Dw * b * p.d1 * p.d2,
            p.Dw * K * p.d2 + p.Dw * b**2 * p.d1 - a * p.alpha * p.d2**2,
            -2.0 * p.alpha * p.d2 * Dscr,
            -p.alpha * b * Dscr,
        ],
        dtype=np.float64,
    )


def chi_of_mu(mu: np.ndarray | float, p: ModelParams) -> np.ndarray | float:
    a = p.r1 + p.k12
    b = p.r2 + p.k21
    K = p.k12 * p.k21
    A = p.d1 * np.asarray(mu) + a
    B = p.d2 * np.asarray(mu) + b
    C = p.Dw * np.asarray(mu) + p.alpha
    value = C * (A * B - K) / (p.beta * np.asarray(mu) * B)
    if np.ndim(mu) == 0:
        return float(value)
    return value


def continuous_critical_data(p: ModelParams) -> Dict[str, float]:
    coeff = quartic_coefficients(p)
    roots = np.roots(coeff)
    positive_real = [
        float(z.real)
        for z in roots
        if abs(z.imag) <= 1.0e-8 * max(1.0, abs(z.real)) and z.real > 0.0
    ]
    if len(positive_real) != 1:
        raise RuntimeError(
            "The quartic should have exactly one positive real root; "
            f"found {positive_real}. Full roots: {roots}"
        )
    mu_c = positive_real[0]
    k_c = math.sqrt(mu_c)
    chi_c = chi_of_mu(mu_c, p)
    Lambda_c = 2.0 * math.pi / k_c
    return {
        "mu_c": mu_c,
        "k_c": k_c,
        "chi_c": chi_c,
        "Lambda_c": Lambda_c,
    }


def select_mode(L: float, p: ModelParams, mu_c: float) -> Dict[str, float | int]:
    k_c = math.sqrt(mu_c)
    xi = L * k_c / math.pi

    if xi < 1.0:
        n_star = 1
    else:
        m = int(math.floor(xi))
        candidates = (max(1, m), m + 1)
        n_star = min(
            candidates,
            key=lambda n: chi_of_mu((n * math.pi / L) ** 2, p),
        )

    mu_star = (n_star * math.pi / L) ** 2
    chi_T = chi_of_mu(mu_star, p)
    return {
        "n_star": int(n_star),
        "mu_star": float(mu_star),
        "k_star": float(math.sqrt(mu_star)),
        "chi_T": float(chi_T),
        "xi": float(xi),
    }


def chi2_data(L: float, p: ModelParams, n_star: int, mu_star: float, chi_T: float) -> Dict[str, float]:
    a = p.r1 + p.k12
    b = p.r2 + p.k21
    K = p.k12 * p.k21

    A = p.d1 * mu_star + a
    B = p.d2 * mu_star + b
    C = p.Dw * mu_star + p.alpha

    eta = p.k21 / B
    zeta = p.beta / C

    Delta0 = a * b - K
    U0 = -(b * p.r1 + p.k12 * p.r2 * eta**2) / (2.0 * Delta0)
    V0 = (p.k21 * U0 - 0.5 * p.r2 * eta**2) / b
    W0 = (p.beta / p.alpha) * U0

    A2 = 4.0 * p.d1 * mu_star + a
    B2 = 4.0 * p.d2 * mu_star + b
    C2 = 4.0 * p.Dw * mu_star + p.alpha

    f_u2 = 0.5 * p.r1 - chi_T * zeta * mu_star
    f_v2 = 0.5 * p.r2 * eta**2

    D2 = A2 - (p.k12 * p.k21) / B2 - 4.0 * chi_T * p.beta * mu_star / C2
    if D2 <= 0.0:
        raise RuntimeError(
            f"Second harmonic is not strictly stable: D2={D2:.6e}. "
            "The simple critical-mode expansion is not valid."
        )

    U2 = -(f_u2 + (p.k12 / B2) * f_v2) / D2
    V2 = (p.k21 * U2 - f_v2) / B2
    W2 = (p.beta / C2) * U2

    N1 = (
        chi_T * mu_star * (W2 + zeta * U0 - 0.5 * zeta * U2)
        - 2.0 * p.r1 * U0
        - p.r1 * U2
    )
    N2 = -2.0 * p.r2 * eta * V0 - p.r2 * eta * V2

    chi2 = -(C / (p.beta * mu_star)) * (N1 + (p.k12 / B) * N2)

    return {
        "A_star": A,
        "B_star": B,
        "C_star": C,
        "eta": eta,
        "zeta": zeta,
        "Delta0": Delta0,
        "U0": U0,
        "V0": V0,
        "W0": W0,
        "A2": A2,
        "B2": B2,
        "C2": C2,
        "D2": D2,
        "f_u2": f_u2,
        "f_v2": f_v2,
        "U2": U2,
        "V2": V2,
        "W2": W2,
        "N1": N1,
        "N2": N2,
        "chi2": chi2,
    }


def dominant_linear_mode(p: ModelParams, mu: float, chi: float) -> Tuple[float, np.ndarray]:
    A = p.d1 * mu + p.r1 + p.k12
    B = p.d2 * mu + p.r2 + p.k21
    C = p.Dw * mu + p.alpha
    Lmat = np.array(
        [
            [-A, p.k12, chi * mu],
            [p.k21, -B, 0.0],
            [p.beta / p.tau, 0.0, -C / p.tau],
        ],
        dtype=np.float64,
    )
    vals, vecs = np.linalg.eig(Lmat)
    idx = int(np.argmax(vals.real))
    lam = float(vals[idx].real)
    vec = np.real(vecs[:, idx])
    if abs(vec[0]) < 1.0e-14:
        raise RuntimeError("Dominant eigenvector has numerically zero u component.")
    vec = vec / vec[0]
    return lam, vec


# ============================================================
# FFT pseudospectral IMEX solver
# ============================================================

class SpectralIMEX1D:
    """
    Even 2L-periodic extension of a Neumann problem on [0,L].

    If n_half = N+1, the full periodic grid has M=2N points and
    dx=L/N. The point x=L is present at index N.
    """

    def __init__(
        self,
        p: ModelParams,
        L: float,
        n_half: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if n_half < 9:
            raise ValueError("n_half is too small.")
        self.p = p
        self.L = float(L)
        self.n_half = int(n_half)
        self.M = 2 * (self.n_half - 1)
        self.dt = float(dt)
        self.device = device
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        self.dx = self.L / (self.n_half - 1)
        self.x = torch.arange(self.M, device=device, dtype=dtype) * self.dx

        freq = torch.fft.fftfreq(self.M, d=self.dx, device=device, dtype=dtype)
        self.k = 2.0 * math.pi * freq
        self.k2 = self.k * self.k

        mode_index = torch.fft.fftfreq(
            self.M, d=1.0 / self.M, device=device, dtype=dtype
        )
        self.dealias = (torch.abs(mode_index) < self.M / 3.0).to(dtype)

        self._build_linear_inverse()

    def _build_linear_inverse(self) -> None:
        p = self.p
        Lm = torch.zeros(
            (self.M, 3, 3), device=self.device, dtype=self.dtype
        )

        # Full (not perturbation) linear part.
        Lm[:, 0, 0] = -p.d1 * self.k2 + (p.r1 - p.k12)
        Lm[:, 0, 1] = p.k12

        Lm[:, 1, 0] = p.k21
        Lm[:, 1, 1] = -p.d2 * self.k2 + (p.r2 - p.k21)

        Lm[:, 2, 0] = p.beta / p.tau
        Lm[:, 2, 2] = -(p.Dw / p.tau) * self.k2 - p.alpha / p.tau

        eye = torch.eye(3, device=self.device, dtype=self.dtype).expand(
            self.M, 3, 3
        )
        self.linear_inverse = torch.linalg.inv(eye - self.dt * Lm).to(self.cdtype)

    def step(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        chi: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.p

        uh = torch.fft.fft(u)
        vh = torch.fft.fft(v)
        wh = torch.fft.fft(w)

        wx = torch.fft.ifft((1j * self.k).to(self.cdtype) * wh).real

        u2h = torch.fft.fft(u * u) * self.dealias
        v2h = torch.fft.fft(v * v) * self.dealias
        fluxh = torch.fft.fft(u * wx) * self.dealias
        flux_x_h = (1j * self.k).to(self.cdtype) * fluxh

        Nu_h = -p.r1 * u2h - chi * flux_x_h
        Nv_h = -p.r2 * v2h

        rhs = torch.stack(
            [
                uh + self.dt * Nu_h,
                vh + self.dt * Nv_h,
                wh,
            ],
            dim=-1,
        )

        zh = torch.einsum("kij,kj->ki", self.linear_inverse, rhs)

        u_new = torch.fft.ifft(zh[:, 0]).real
        v_new = torch.fft.ifft(zh[:, 1]).real
        w_new = torch.fft.ifft(zh[:, 2]).real
        return u_new, v_new, w_new

    def cosine_amplitude(self, field: torch.Tensor, n: int, mean: float) -> float:
        fh = torch.fft.fft(field - mean)
        # On the 2L-periodic extension:
        # mean + A cos(n*pi*x/L) -> FFT coefficient M*A/2 at mode n.
        amp = 2.0 * fh[n].real / self.M
        return float(amp.detach().cpu())

    def steady_residual(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        chi: float,
    ) -> float:
        p = self.p

        uh = torch.fft.fft(u)
        vh = torch.fft.fft(v)
        wh = torch.fft.fft(w)

        uxx = torch.fft.ifft((-self.k2).to(self.cdtype) * uh).real
        vxx = torch.fft.ifft((-self.k2).to(self.cdtype) * vh).real
        wxx = torch.fft.ifft((-self.k2).to(self.cdtype) * wh).real
        wx = torch.fft.ifft((1j * self.k).to(self.cdtype) * wh).real

        fluxh = torch.fft.fft(u * wx) * self.dealias
        fluxx = torch.fft.ifft((1j * self.k).to(self.cdtype) * fluxh).real

        ru = (
            p.d1 * uxx
            - chi * fluxx
            + p.r1 * u * (1.0 - u)
            + p.k12 * (v - u)
        )
        rv = (
            p.d2 * vxx
            + p.r2 * v * (1.0 - v)
            + p.k21 * (u - v)
        )
        rw = (p.Dw * wxx - p.alpha * w + p.beta * u) / p.tau

        max_res = torch.max(
            torch.stack([ru.abs().max(), rv.abs().max(), rw.abs().max()])
        )
        return float(max_res.detach().cpu())


# ============================================================
# Numerical experiments
# ============================================================

def fit_log_growth_rate(t: np.ndarray, a: np.ndarray, a0: float) -> float:
    mask = (
        np.isfinite(a)
        & (a > max(1.0e-15, 0.35 * a0))
        & (a < 3.0 * a0)
    )
    if np.count_nonzero(mask) < 6:
        mask = np.isfinite(a) & (a > 1.0e-15)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return float(np.polyfit(t[mask], np.log(a[mask]), 1)[0])


def actual_imex_growth_rate(p, mu, chi, dt):
    """Growth rate from the amplification matrix of this actual splitting."""
    implicit = np.array([
        [-p.d1*mu+p.r1-p.k12, p.k12, 0.0],
        [p.k21, -p.d2*mu+p.r2-p.k21, 0.0],
        [p.beta/p.tau, 0.0, -(p.Dw*mu+p.alpha)/p.tau],
    ], dtype=float)
    explicit = np.array([[-2*p.r1, 0.0, chi*mu],
                         [0.0, -2*p.r2, 0.0], [0.0, 0.0, 0.0]])
    G = np.linalg.solve(np.eye(3)-dt*implicit, np.eye(3)+dt*explicit)
    return float(np.log(np.max(np.abs(np.linalg.eigvals(G))))/dt)


def threshold_crossing_run(
    solver: SpectralIMEX1D,
    p: ModelParams,
    n_star: int,
    mu_star: float,
    chi_T: float,
    ratio: float,
    tmax: float,
    eps: float,
    sample_dt: float,
) -> Dict[str, object]:
    chi = ratio * chi_T
    lam_cont, eigenvector = dominant_linear_mode(p, mu_star, chi)

    lam_imex = actual_imex_growth_rate(p, mu_star, chi, solver.dt)

    mode = torch.cos(n_star * math.pi * solver.x / solver.L)
    u = 1.0 + eps * eigenvector[0] * mode
    v = 1.0 + eps * eigenvector[1] * mode
    w = p.beta / p.alpha + eps * eigenvector[2] * mode

    steps = int(math.ceil(tmax / solver.dt))
    sample_every = max(1, int(round(sample_dt / solver.dt)))

    times: List[float] = []
    amplitudes: List[float] = []

    for step in range(steps + 1):
        if step % sample_every == 0:
            times.append(step * solver.dt)
            amplitudes.append(
                abs(solver.cosine_amplitude(u, n_star, mean=1.0))
            )
        if step < steps:
            u, v, w = solver.step(u, v, w, chi)

    times_np = np.asarray(times, dtype=np.float64)
    amp_np = np.asarray(amplitudes, dtype=np.float64)
    fit_rate = fit_log_growth_rate(times_np, amp_np, eps)

    return {
        "ratio": ratio,
        "chi": chi,
        "lambda_continuous": lam_cont,
        "lambda_imex": lam_imex,
        "fit_rate": fit_rate,
        "times": times_np,
        "amplitudes": amp_np,
        "eigenvector": eigenvector,
    }


def initialize_branch_state(
    solver: SpectralIMEX1D,
    p: ModelParams,
    n_star: int,
    chi2: float,
    delta_chi: float,
    c2: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if chi2 <= 0.0:
        raise RuntimeError(
            "The time-relaxation branch validation assumes chi2>0 "
            "(locally stable right-going branch)."
        )

    A = math.sqrt(delta_chi / chi2)
    c1mode = torch.cos(n_star * math.pi * solver.x / solver.L)
    c2mode = torch.cos(2 * n_star * math.pi * solver.x / solver.L)

    u = (
        1.0
        + A * c1mode
        + A * A * (c2["U0"] + c2["U2"] * c2mode)
    )
    v = (
        1.0
        + A * c2["eta"] * c1mode
        + A * A * (c2["V0"] + c2["V2"] * c2mode)
    )
    w = (
        p.beta / p.alpha
        + A * c2["zeta"] * c1mode
        + A * A * (c2["W0"] + c2["W2"] * c2mode)
    )
    return u, v, w, A


def stationary_branch_run(
    solver: SpectralIMEX1D,
    p: ModelParams,
    n_star: int,
    chi_T: float,
    chi2: float,
    c2: Dict[str, float],
    delta_chi: float,
    tmax: float,
    tolerance: float,
    check_dt: float,
) -> Dict[str, object]:
    chi = chi_T + delta_chi
    u, v, w, A_prediction = initialize_branch_state(
        solver, p, n_star, chi2, delta_chi, c2
    )

    steps = int(math.ceil(tmax / solver.dt))
    check_every = max(1, int(round(check_dt / solver.dt)))

    residual = float("inf")
    elapsed_sim = 0.0

    for step in range(steps):
        u, v, w = solver.step(u, v, w, chi)

        if (step + 1) % check_every == 0:
            elapsed_sim = (step + 1) * solver.dt
            residual = solver.steady_residual(u, v, w, chi)
            if residual < tolerance and elapsed_sim >= 5.0:
                break

    # Recompute diagnostics at the actual final state even if no check fired.
    elapsed_sim = (step + 1) * solver.dt
    residual = solver.steady_residual(u, v, w, chi)
    converged = bool(residual < tolerance and elapsed_sim >= 5.0)
    amplitude = solver.cosine_amplitude(u, n_star, mean=1.0)

    # Restrict profile to [0,L], including both endpoints.
    stop = solver.n_half
    x_half = solver.x[:stop].detach().cpu().numpy()
    u_half = u[:stop].detach().cpu().numpy()
    v_half = v[:stop].detach().cpu().numpy()
    w_half = w[:stop].detach().cpu().numpy()

    return {
        "delta_chi": delta_chi,
        "chi": chi,
        "A_prediction": A_prediction,
        "A_numeric": amplitude,
        "A2_numeric": amplitude * amplitude,
        "residual": residual,
        "sim_time": elapsed_sim,
        "residual_tolerance_met": converged,
        "x": x_half,
        "u": u_half,
        "v": v_half,
        "w": w_half,
    }


# ============================================================
# Output helpers
# ============================================================

def ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_dispersion_plot(
    out: Path,
    p: ModelParams,
    continuous: Dict[str, float],
    selected: Dict[str, float | int],
    cfg: RunConfig,
) -> None:
    mu = np.linspace(cfg.mu_plot_min, cfg.mu_plot_max, cfg.mu_plot_points)
    vals = chi_of_mu(mu, p)

    plt.figure(figsize=(7.2, 5.0))
    plt.plot(mu, vals, label=r"$\chi(\mu)$")
    plt.scatter(
        [continuous["mu_c"]],
        [continuous["chi_c"]],
        label=r"continuous minimum",
    )

    nmax = max(8, int(2.0 * cfg.L * continuous["k_c"] / math.pi) + 5)
    mus = np.array([(n * math.pi / cfg.L) ** 2 for n in range(1, nmax + 1)])
    chis = np.array([chi_of_mu(x, p) for x in mus])
    plt.scatter(mus, chis, label=f"Neumann modes, L={cfg.L:g}")

    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\chi(\mu)$")
    plt.title("Continuous threshold and discrete Neumann modes")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "dispersion_curve.png", dpi=220)
    plt.close()


def save_mode_selection_data_and_plots(
    out: Path,
    p: ModelParams,
    continuous: Dict[str, float],
    cfg: RunConfig,
) -> None:
    Ls = np.linspace(cfg.L_min, cfg.L_max, cfg.L_points)
    rows = []
    for L in Ls:
        s = select_mode(float(L), p, continuous["mu_c"])
        rows.append(
            {
                "L": float(L),
                "n_star": int(s["n_star"]),
                "mu_star": float(s["mu_star"]),
                "k_star": float(s["k_star"]),
                "chi_T": float(s["chi_T"]),
                "chi_T_minus_chi_c": float(s["chi_T"] - continuous["chi_c"]),
            }
        )

    write_csv(
        out / "mode_selection.csv",
        rows,
        ["L", "n_star", "mu_star", "k_star", "chi_T", "chi_T_minus_chi_c"],
    )

    Larr = np.array([r["L"] for r in rows])
    narr = np.array([r["n_star"] for r in rows])
    kt = Larr * continuous["k_c"] / math.pi

    plt.figure(figsize=(7.2, 5.0))
    plt.plot(Larr, narr, label=r"$n_*(L)$")
    plt.plot(Larr, kt, label=r"$Lk_c/\pi$")
    plt.xlabel(r"$L$")
    plt.ylabel("selected mode")
    plt.title("Finite-domain mode-selection law")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "mode_selection.png", dpi=220)
    plt.close()

    err = np.maximum(
        np.array([r["chi_T_minus_chi_c"] for r in rows]),
        np.finfo(float).tiny,
    )
    plt.figure(figsize=(7.2, 5.0))
    plt.loglog(Larr, err, label=r"$\chi_T(L)-\chi_c$")
    ref = err[-1] * (Larr[-1] / Larr) ** 2
    plt.loglog(Larr, ref, label=r"$L^{-2}$ reference")
    plt.xlabel(r"$L$")
    plt.ylabel(r"$\chi_T(L)-\chi_c$")
    plt.title("Finite-size threshold error")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "threshold_finite_size.png", dpi=220)
    plt.close()


def save_crossing_outputs(out: Path, runs: List[Dict[str, object]]) -> None:
    rows = []
    for run in runs:
        for t, a in zip(run["times"], run["amplitudes"]):
            rows.append(
                {
                    "ratio": run["ratio"],
                    "chi": run["chi"],
                    "time": float(t),
                    "amplitude": float(a),
                    "lambda_continuous": run["lambda_continuous"],
                    "lambda_imex": run["lambda_imex"],
                    "fit_rate": run["fit_rate"],
                }
            )

    write_csv(
        out / "threshold_crossing.csv",
        rows,
        [
            "ratio",
            "chi",
            "time",
            "amplitude",
            "lambda_continuous",
            "lambda_imex",
            "fit_rate",
        ],
    )

    plt.figure(figsize=(7.2, 5.0))
    for run in runs:
        plt.semilogy(
            run["times"],
            run["amplitudes"],
            label=(
                f"chi/chiT={run['ratio']:.3f}, "
                f"fit={run['fit_rate']:.4e}"
            ),
        )
    plt.xlabel("time")
    plt.ylabel("critical-mode amplitude")
    plt.title("Below/above-threshold modal dynamics")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "threshold_crossing.png", dpi=220)
    plt.close()


def save_branch_outputs(
    out: Path,
    branch_runs: List[Dict[str, object]],
    chi2_theory: float,
) -> Dict[str, float]:
    rows = []
    for r in branch_runs:
        rows.append(
            {
                "delta_chi": r["delta_chi"],
                "chi": r["chi"],
                "A_prediction": r["A_prediction"],
                "A_numeric": r["A_numeric"],
                "A2_numeric": r["A2_numeric"],
                "delta_over_A2": (
                    r["delta_chi"] / r["A2_numeric"]
                    if r["A2_numeric"] > 0.0
                    else float("nan")
                ),
                "residual": r["residual"],
                "sim_time": r["sim_time"],
                "residual_tolerance_met": r["residual_tolerance_met"],
            }
        )

    write_csv(
        out / "branch_results.csv",
        rows,
        [
            "delta_chi",
            "chi",
            "A_prediction",
            "A_numeric",
            "A2_numeric",
            "delta_over_A2",
            "residual",
            "sim_time",
            "residual_tolerance_met",
        ],
    )

    x = np.array([r["A2_numeric"] for r in branch_runs], dtype=float)
    y = np.array([r["delta_chi"] for r in branch_runs], dtype=float)

    slope_origin = float(np.dot(x, y) / np.dot(x, x))
    slope_free, intercept = np.polyfit(x, y, 1)

    xline = np.linspace(0.0, max(x) * 1.05, 160)

    plt.figure(figsize=(7.2, 5.0))
    plt.scatter(x, y, label="terminal-time states")
    plt.plot(xline, chi2_theory * xline, label="theory: chi2 A^2")
    plt.plot(xline, slope_origin * xline, label="fit through origin")
    plt.xlabel(r"$A^2$")
    plt.ylabel(r"$\chi-\chi_T$")
    plt.title("Nonlinear branch-slope validation")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "branch_slope.png", dpi=220)
    plt.close()

    # Final profile at the largest delta.
    last = branch_runs[-1]
    plt.figure(figsize=(7.2, 5.0))
    plt.plot(last["x"], last["u"], label="u")
    plt.plot(last["x"], last["v"], label="v")
    plt.plot(last["x"], last["w"], label="w")
    plt.xlabel("x")
    plt.ylabel("terminal profile")
    plt.title(
        f"Terminal pattern, delta_chi={last['delta_chi']:.4g}"
    )
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "branch_profile.png", dpi=220)
    plt.close()

    return {
        "branch_slope_origin": slope_origin,
        "branch_slope_free": float(slope_free),
        "branch_intercept_free": float(intercept),
        "branch_relative_error_origin": abs(slope_origin - chi2_theory)
        / abs(chi2_theory),
    }


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or "CPU"


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=["quick", "paper", "stress"],
        default="quick",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="Default: float64 on CPU and CUDA for near-onset diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_task16_kpp_chemotaxis",
    )
    parser.add_argument("--skip-crossing", action="store_true")
    parser.add_argument("--skip-branch", action="store_true")
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = preset_config(args.preset)
    p = ModelParams()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dtype is None:
        dtype = torch.float64
    else:
        dtype = torch.float32 if args.dtype == "float32" else torch.float64

    out = ensure_output_dir(args.output_dir)

    continuous = continuous_critical_data(p)
    selected = select_mode(cfg.L, p, continuous["mu_c"])
    c2 = chi2_data(
        cfg.L,
        p,
        int(selected["n_star"]),
        float(selected["mu_star"]),
        float(selected["chi_T"]),
    )

    print("=" * 72)
    print("Task 1.6B — cooperative Fisher-KPP-chemotaxis validation")
    print("=" * 72)
    print(f"device       : {device} ({device_description(device)})")
    print(f"dtype        : {dtype}")
    print(f"preset       : {cfg.preset}")
    print(f"L            : {cfg.L}")
    print(f"n_half       : {cfg.n_half}")
    print(f"full FFT M   : {2 * (cfg.n_half - 1)}")
    print("-" * 72)
    print(f"mu_c         : {continuous['mu_c']:.12f}")
    print(f"k_c          : {continuous['k_c']:.12f}")
    print(f"Lambda_c     : {continuous['Lambda_c']:.12f}")
    print(f"chi_c        : {continuous['chi_c']:.12f}")
    print(f"n_star(L)    : {selected['n_star']}")
    print(f"mu_star(L)   : {selected['mu_star']:.12f}")
    print(f"chi_T(L)     : {selected['chi_T']:.12f}")
    print(f"chi2 theory  : {c2['chi2']:.12f}")
    print(f"D2           : {c2['D2']:.12f}")
    print("=" * 72)

    save_dispersion_plot(out, p, continuous, selected, cfg)
    save_mode_selection_data_and_plots(out, p, continuous, cfg)

    summary: Dict[str, object] = {
        "model": asdict(p),
        "run_config": asdict(cfg),
        "device": str(device),
        "device_name": device_description(device),
        "dtype": str(dtype),
        "continuous": continuous,
        "selected": selected,
        "chi2_data": c2,
        "torch_version": torch.__version__,
    }

    wall_start = time.perf_counter()

    # --------------------------------------------------------
    # Threshold crossing
    # --------------------------------------------------------
    if not args.skip_crossing:
        solver_cross = SpectralIMEX1D(
            p=p,
            L=cfg.L,
            n_half=cfg.n_half,
            dt=cfg.dt_cross,
            device=device,
            dtype=dtype,
        )

        crossing_runs = []
        for ratio in cfg.ratios:
            print(f"[crossing] chi/chiT={ratio:.4f} ...")
            run = threshold_crossing_run(
                solver=solver_cross,
                p=p,
                n_star=int(selected["n_star"]),
                mu_star=float(selected["mu_star"]),
                chi_T=float(selected["chi_T"]),
                ratio=ratio,
                tmax=cfg.t_cross,
                eps=cfg.eps_cross,
                sample_dt=cfg.sample_dt,
            )
            crossing_runs.append(run)
            print(
                "  continuous lambda = "
                f"{run['lambda_continuous']:+.8e}, "
                "IMEX lambda = "
                f"{run['lambda_imex']:+.8e}, "
                "fit = "
                f"{run['fit_rate']:+.8e}"
            )

        save_crossing_outputs(out, crossing_runs)

        summary["threshold_crossing"] = [
            {
                "ratio": r["ratio"],
                "chi": r["chi"],
                "lambda_continuous": r["lambda_continuous"],
                "lambda_imex": r["lambda_imex"],
                "fit_rate": r["fit_rate"],
                "eigenvector": np.asarray(r["eigenvector"]).tolist(),
            }
            for r in crossing_runs
        ]

    # --------------------------------------------------------
    # Nonlinear stationary branch
    # --------------------------------------------------------
    if not args.skip_branch:
        if c2["chi2"] <= 0.0:
            print(
                "[branch] chi2 <= 0, so forward-time relaxation is not "
                "appropriate for the locally unstable branch. Skipping."
            )
        else:
            solver_branch = SpectralIMEX1D(
                p=p,
                L=cfg.L,
                n_half=cfg.n_half,
                dt=cfg.dt_branch,
                device=device,
                dtype=dtype,
            )

            branch_runs = []
            for delta in cfg.branch_deltas:
                print(f"[branch] delta_chi={delta:.6g} ...")
                run = stationary_branch_run(
                    solver=solver_branch,
                    p=p,
                    n_star=int(selected["n_star"]),
                    chi_T=float(selected["chi_T"]),
                    chi2=float(c2["chi2"]),
                    c2=c2,
                    delta_chi=float(delta),
                    tmax=cfg.t_branch,
                    tolerance=cfg.branch_tol,
                    check_dt=cfg.branch_check_dt,
                )
                branch_runs.append(run)
                ratio = (
                    run["delta_chi"] / run["A2_numeric"]
                    if run["A2_numeric"] > 0.0
                    else float("nan")
                )
                print(
                    f"  A_pred={run['A_prediction']:.8e}, "
                    f"A_num={run['A_numeric']:.8e}, "
                    f"delta/A^2={ratio:.8f}, "
                    f"residual={run['residual']:.3e}, "
                    f"T={run['sim_time']:.2f}"
                )

            fit_data = save_branch_outputs(
                out, branch_runs, chi2_theory=float(c2["chi2"])
            )
            summary["branch_fit"] = fit_data
            summary["branch_runs"] = [
                {
                    "delta_chi": r["delta_chi"],
                    "chi": r["chi"],
                    "A_prediction": r["A_prediction"],
                    "A_numeric": r["A_numeric"],
                    "A2_numeric": r["A2_numeric"],
                    "residual": r["residual"],
                    "sim_time": r["sim_time"],
                    "residual_tolerance_met": r["residual_tolerance_met"],
                }
                for r in branch_runs
            ]

            print("-" * 72)
            print(
                "branch slope through origin : "
                f"{fit_data['branch_slope_origin']:.10f}"
            )
            print(
                "chi2 theory                 : "
                f"{c2['chi2']:.10f}"
            )
            print(
                "relative slope error         : "
                f"{fit_data['branch_relative_error_origin']:.3e}"
            )

    elapsed = time.perf_counter() - wall_start
    summary["wall_seconds"] = elapsed

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print(f"Finished in {elapsed:.2f} s")
    print(f"Results saved to: {out.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
