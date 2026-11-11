#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 1.8A — High-end publication figures
Cooperative Fisher-KPP-chemotaxis system

The script preserves all analytical and numerical conclusions while
upgrading the graphical presentation to a modern PDE / nonlinear
dynamics journal style.

Generated figures
-----------------
1. figure1_dispersion.png / pdf
2. figure2_threshold_growth.png / pdf
3. figure3_time_convergence.png / pdf
4. figure4_bifurcation_branch.png / pdf
5. figure_task18_overview.png / pdf

Required input
--------------
results_task17/
    time_convergence.csv

results_task17B_advanced/
    branch_continuation_advanced.csv
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ============================================================
# Paths
# ============================================================

OUT = Path("figures_task18")
OUT.mkdir(exist_ok=True)

TIME_FILE = Path(
    "results_task17/time_convergence.csv"
)

BRANCH_FILE = Path(
    "results_task17B_advanced/"
    "branch_continuation_advanced.csv"
)


# ============================================================
# Model parameters
# ============================================================

d1 = 0.2
d2 = 0.1
Dw = 1.0

r1 = 1.0
r2 = 0.8

k12 = 0.3
k21 = 0.2

alpha = 1.0
beta = 1.0

L = 10.0

mu_c = 2.468635740034
chi_c = 2.452720209264

chi2 = 2.1143925003845543

lambda_below = -2.27279492e-2
lambda_above = 2.25494207e-2

critical_amplitude_0 = 1.0e-4


# ============================================================
# Global journal-style configuration
# ============================================================

plt.rcParams.update({

    # typography
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "STIXGeneral",
        "DejaVu Serif"
    ],
    "mathtext.fontset": "stix",

    "font.size": 10.5,
    "axes.labelsize": 12.5,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,

    # axes
    "axes.linewidth": 0.85,
    "axes.spines.top": False,
    "axes.spines.right": False,

    # ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    # lines
    "lines.linewidth": 2.0,

    # saving
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


# ============================================================
# Palette
#
# Deliberately restrained:
# no rainbow maps, no unnecessary gradients.
# ============================================================

BLUE = "#2166AC"
ORANGE = "#D95F02"
RED = "#B2182B"
GREEN = "#1B7837"
DARK = "#222222"
GREY = "#737373"
LIGHT_GREY = "#D9D9D9"


# ============================================================
# Utility functions
# ============================================================

def read_csv(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path.resolve()}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_figure(fig, basename):

    png_path = OUT / f"{basename}.png"
    pdf_path = OUT / f"{basename}.pdf"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )

    print("Saved:", png_path.resolve())
    print("Saved:", pdf_path.resolve())


def panel_label(ax, label):

    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left"
    )


# ============================================================
# Exact dispersion relation
# ============================================================

def chi_of_mu(mu):

    a = r1 + k12
    b = r2 + k21

    numerator = (
        (Dw * mu + alpha)
        *
        (
            (d1 * mu + a)
            *
            (d2 * mu + b)
            -
            k12 * k21
        )
    )

    denominator = (
        beta
        *
        mu
        *
        (d2 * mu + b)
    )

    return numerator / denominator


# ============================================================
# FIGURE 1
# Exact dispersion curve and discrete Neumann selection
# ============================================================

def plot_dispersion(ax, panel=None):

    mu = np.linspace(
        0.05,
        8.0,
        3000
    )

    chi = chi_of_mu(mu)

    n_values = np.arange(1, 9)

    mu_n = (
        n_values
        *
        np.pi
        /
        L
    ) ** 2

    chi_n = chi_of_mu(mu_n)

    idx = 4

    # --------------------------------------------------------
    # Main continuous threshold curve
    # --------------------------------------------------------

    ax.plot(
        mu,
        chi,
        color=BLUE,
        lw=2.4,
        zorder=2
    )

    # very subtle area emphasis
    ax.fill_between(
        mu,
        chi_c,
        chi,
        where=(chi >= chi_c),
        color=BLUE,
        alpha=0.055,
        linewidth=0
    )

    # --------------------------------------------------------
    # Neumann modes
    # --------------------------------------------------------

    ax.scatter(
        mu_n,
        chi_n,
        s=35,
        color=GREEN,
        edgecolor="white",
        linewidth=0.55,
        zorder=5
    )

    # selected discrete mode
    ax.scatter(
        mu_n[idx],
        chi_n[idx],
        s=92,
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=1.8,
        zorder=7
    )

    ax.scatter(
        mu_n[idx],
        chi_n[idx],
        s=36,
        color=GREEN,
        zorder=8
    )

    # continuous critical point
    ax.scatter(
        mu_c,
        chi_c,
        s=29,
        color=ORANGE,
        zorder=9
    )

    # --------------------------------------------------------
    # Mode locking guide
    # --------------------------------------------------------

    ax.axvline(
        mu_c,
        color=ORANGE,
        ls=(0, (3, 3)),
        lw=0.9,
        alpha=0.65,
        zorder=1
    )

    # --------------------------------------------------------
    # Annotation
    # --------------------------------------------------------

    ax.annotate(
        r"$\mu_c=2.4686$"
        "\n"
        r"$\chi_c=2.4527$",
        xy=(mu_c, chi_c),
        xytext=(3.55, 6.4),
        fontsize=10.5,
        color=DARK,
        arrowprops=dict(
            arrowstyle="-",
            color=DARK,
            lw=0.9
        )
    )

    ax.annotate(
        r"$n_*=5$",
        xy=(
            mu_n[idx],
            chi_n[idx]
        ),
        xytext=(
            3.25,
            3.6
        ),
        fontsize=11,
        arrowprops=dict(
            arrowstyle="-",
            color=DARK,
            lw=0.9
        )
    )

    # --------------------------------------------------------
    # Inset: critical neighbourhood
    # --------------------------------------------------------

    axins = ax.inset_axes(
        [0.61, 0.56, 0.33, 0.31]
    )

    zoom_mu = np.linspace(
        2.25,
        2.75,
        800
    )

    axins.plot(
        zoom_mu,
        chi_of_mu(zoom_mu),
        color=BLUE,
        lw=1.8
    )

    axins.scatter(
        mu_c,
        chi_c,
        s=24,
        color=ORANGE,
        zorder=6
    )

    axins.scatter(
        mu_n[idx],
        chi_n[idx],
        s=24,
        color=GREEN,
        zorder=7
    )

    axins.axvline(
        mu_c,
        color=ORANGE,
        lw=0.7,
        ls="--",
        alpha=0.5
    )

    axins.set_xlim(
        2.25,
        2.75
    )

    axins.set_ylim(
        2.44,
        2.50
    )

    axins.set_title(
        "critical neighbourhood",
        fontsize=8.5,
        pad=4
    )

    axins.tick_params(
        labelsize=7.5,
        width=0.65,
        length=3
    )

    axins.spines["top"].set_visible(True)
    axins.spines["right"].set_visible(True)

    for spine in axins.spines.values():
        spine.set_linewidth(0.7)

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        r"$\mu$"
    )

    ax.set_ylabel(
        r"$\chi(\mu)$"
    )

    ax.set_xlim(
        0,
        8.0
    )

    ax.set_ylim(
        1.25,
        26.5
    )

    # very restrained guide lines
    ax.grid(
        axis="y",
        color=LIGHT_GREY,
        lw=0.6,
        alpha=0.35
    )

    if panel is not None:
        panel_label(ax, panel)


def figure_dispersion():

    fig, ax = plt.subplots(
        figsize=(7.1, 4.9),
        constrained_layout=True
    )

    plot_dispersion(ax)

    save_figure(
        fig,
        "figure1_dispersion"
    )

    plt.close(fig)


# ============================================================
# FIGURE 2
# Threshold growth / decay
# ============================================================

def plot_threshold_growth(ax, panel=None):

    t = np.linspace(
        0,
        80,
        500
    )

    decay = (
        critical_amplitude_0
        *
        np.exp(
            lambda_below * t
        )
    )

    growth = (
        critical_amplitude_0
        *
        np.exp(
            lambda_above * t
        )
    )

    # threshold reference
    ax.axhline(
        critical_amplitude_0,
        color=GREY,
        lw=0.9,
        ls=(0, (3, 3)),
        alpha=0.7
    )

    ax.semilogy(
        t,
        decay,
        color=BLUE,
        lw=2.3
    )

    ax.semilogy(
        t,
        growth,
        color=RED,
        lw=2.3
    )

    # endpoints
    ax.scatter(
        [t[-1]],
        [decay[-1]],
        s=28,
        color=BLUE,
        zorder=5
    )

    ax.scatter(
        [t[-1]],
        [growth[-1]],
        s=28,
        color=RED,
        zorder=5
    )

    # direct labels rather than a large legend
    ax.text(
        53,
        decay[
            np.searchsorted(t, 53)
        ] * 0.90,
        r"$\chi=0.98\chi_T$",
        color=BLUE,
        fontsize=10
    )

    ax.text(
        51,
        growth[
            np.searchsorted(t, 51)
        ] * 1.10,
        r"$\chi=1.02\chi_T$",
        color=RED,
        fontsize=10
    )

    ax.text(
        4,
        1.07e-4,
        r"$A(0)=10^{-4}$",
        color=GREY,
        fontsize=9
    )

    ax.text(
        5,
        5.4e-5,
        r"$\lambda_-=-2.2728\times10^{-2}$",
        color=BLUE,
        fontsize=9.2
    )

    ax.text(
        5,
        1.72e-4,
        r"$\lambda_+=2.2549\times10^{-2}$",
        color=RED,
        fontsize=9.2
    )

    ax.set_xlabel(
        r"$t$"
    )

    ax.set_ylabel(
        r"critical-mode amplitude"
    )

    ax.set_xlim(
        0,
        80
    )

    ax.grid(
        axis="y",
        which="major",
        color=LIGHT_GREY,
        alpha=0.42,
        lw=0.65
    )

    if panel is not None:
        panel_label(ax, panel)


def figure_threshold_growth():

    fig, ax = plt.subplots(
        figsize=(7.0, 4.7),
        constrained_layout=True
    )

    plot_threshold_growth(ax)

    save_figure(
        fig,
        "figure2_threshold_growth"
    )

    plt.close(fig)


# ============================================================
# FIGURE 3
# Temporal convergence
# ============================================================

def get_time_data():

    rows = read_csv(
        TIME_FILE
    )

    dt = np.array(
        [
            float(r["dt"])
            for r in rows
        ]
    )

    error = np.array(
        [
            float(r["absolute_error"])
            for r in rows
        ]
    )

    order = np.argsort(dt)

    return (
        dt[order],
        error[order]
    )


def plot_time_convergence(ax, panel=None):

    dt, error = get_time_data()

    ax.loglog(
        dt,
        error,
        color=BLUE,
        lw=2.0,
        marker="o",
        markersize=6.2,
        markerfacecolor="white",
        markeredgewidth=1.4,
        markeredgecolor=BLUE,
        zorder=4
    )

    # --------------------------------------------------------
    # First-order reference slope
    # Anchored to smallest dt result.
    # --------------------------------------------------------

    dt_ref = np.linspace(
        dt.min(),
        dt.max(),
        200
    )

    C = (
        error[0]
        /
        dt[0]
    )

    first_order = (
        C * dt_ref
    )

    ax.loglog(
        dt_ref,
        first_order,
        color=GREY,
        lw=1.1,
        ls="--",
        label=r"$O(\Delta t)$"
    )

    # empirical least-squares slope
    p = np.polyfit(
        np.log(dt),
        np.log(error),
        1
    )[0]

    ax.text(
        0.05,
        0.91,
        rf"observed slope $p={p:.3f}$",
        transform=ax.transAxes,
        fontsize=10,
        va="top"
    )

    ax.text(
        0.05,
        0.82,
        r"reference: $O(\Delta t)$",
        transform=ax.transAxes,
        fontsize=9.3,
        color=GREY,
        va="top"
    )

    ax.set_xlabel(
        r"time step $\Delta t$"
    )

    ax.set_ylabel(
        r"$E_\lambda$"
    )

    ax.grid(
        which="major",
        color=LIGHT_GREY,
        alpha=0.40,
        lw=0.65
    )

    ax.grid(
        which="minor",
        color=LIGHT_GREY,
        alpha=0.16,
        lw=0.45
    )

    if panel is not None:
        panel_label(ax, panel)


def figure_time_convergence():

    fig, ax = plt.subplots(
        figsize=(6.7, 4.7),
        constrained_layout=True
    )

    plot_time_convergence(ax)

    save_figure(
        fig,
        "figure3_time_convergence"
    )

    plt.close(fig)


# ============================================================
# FIGURE 4
# Weakly nonlinear branch
# ============================================================

def get_branch_data():

    rows = read_csv(
        BRANCH_FILE
    )

    dchi = np.array(
        [
            float(r["delta_chi"])
            for r in rows
        ]
    )

    A = np.array(
        [
            float(r["A_numeric"])
            for r in rows
        ]
    )

    order = np.argsort(dchi)

    return (
        dchi[order],
        A[order]
    )


def plot_branch(ax, panel=None):

    dchi, A = get_branch_data()

    A2 = A ** 2

    theory_at_data = (
        dchi
        /
        chi2
    )

    xx = np.linspace(
        0,
        dchi.max() * 1.10,
        500
    )

    theory = (
        xx
        /
        chi2
    )

    # --------------------------------------------------------
    # Theory band
    # --------------------------------------------------------

    ax.plot(
        xx,
        theory,
        color=DARK,
        lw=1.7,
        zorder=2
    )

    # --------------------------------------------------------
    # Numerical continuation
    # --------------------------------------------------------

    ax.plot(
        dchi,
        A2,
        color=BLUE,
        lw=1.2,
        alpha=0.55,
        zorder=3
    )

    ax.scatter(
        dchi,
        A2,
        s=48,
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.75,
        zorder=5
    )

    # threshold point
    ax.scatter(
        [0],
        [0],
        s=33,
        facecolor=ORANGE,
        edgecolor="white",
        linewidth=0.6,
        zorder=6
    )

    # --------------------------------------------------------
    # Direct labels
    # --------------------------------------------------------

    ax.text(
        0.00655,
        0.00320,
        r"theory:"
        "\n"
        r"$A^2=(\chi-\chi_T)/\chi_2$",
        color=DARK,
        fontsize=9.5
    )

    ax.text(
        0.0018,
        0.00165,
        "numerical branch",
        color=BLUE,
        fontsize=9.6
    )

    ax.text(
        0.00035,
        0.00010,
        r"$\chi=\chi_T$",
        fontsize=9.1,
        color=ORANGE
    )

    # --------------------------------------------------------
    # Inset:
    # relative discrepancy from leading-order branch
    # --------------------------------------------------------

    inset = ax.inset_axes(
        [0.57, 0.13, 0.36, 0.30]
    )

    discrepancy = (
        (
            A2 - theory_at_data
        )
        /
        theory_at_data
        *
        100.0
    )

    inset.plot(
        dchi,
        discrepancy,
        color=ORANGE,
        marker="o",
        markersize=4.3,
        lw=1.2
    )

    inset.axhline(
        0,
        color=GREY,
        lw=0.7,
        ls="--"
    )

    inset.set_xlabel(
        r"$\Delta\chi$",
        fontsize=7.5
    )

    inset.set_ylabel(
        "deviation (%)",
        fontsize=7.5
    )

    inset.set_title(
        "higher-order departure",
        fontsize=8.2,
        pad=3
    )

    inset.tick_params(
        labelsize=7,
        width=0.6,
        length=2.5
    )

    inset.spines["top"].set_visible(True)
    inset.spines["right"].set_visible(True)

    for spine in inset.spines.values():
        spine.set_linewidth(0.65)

    # --------------------------------------------------------
    # Main axes
    # --------------------------------------------------------

    ax.set_xlabel(
        r"$\Delta\chi=\chi-\chi_T$"
    )

    ax.set_ylabel(
        r"$A_{\rm num}^{\,2}$"
    )

    ax.set_xlim(
        -0.0002,
        dchi.max() * 1.13
    )

    ymax = max(
        theory.max(),
        A2.max()
    )

    ax.set_ylim(
        -0.00015 * ymax,
        ymax * 1.08
    )

    ax.grid(
        color=LIGHT_GREY,
        alpha=0.35,
        lw=0.65
    )

    if panel is not None:
        panel_label(ax, panel)


def figure_branch():

    fig, ax = plt.subplots(
        figsize=(7.1, 4.9),
        constrained_layout=True
    )

    plot_branch(ax)

    save_figure(
        fig,
        "figure4_bifurcation_branch"
    )

    plt.close(fig)


# ============================================================
# Composite four-panel overview
# ============================================================

def figure_overview():

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.2, 8.4),
        constrained_layout=True
    )

    plot_dispersion(
        axes[0, 0],
        panel="a"
    )

    plot_threshold_growth(
        axes[0, 1],
        panel="b"
    )

    plot_time_convergence(
        axes[1, 0],
        panel="c"
    )

    plot_branch(
        axes[1, 1],
        panel="d"
    )

    save_figure(
        fig,
        "figure_task18_overview"
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Task 1.8A — publication-quality figure generation")
    print("=" * 70)

    figure_dispersion()

    figure_threshold_growth()

    figure_time_convergence()

    figure_branch()

    figure_overview()

    print()
    print("=" * 70)
    print("Task 1.8A figures finished.")
    print("Saved in:")
    print(OUT.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()