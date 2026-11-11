#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 1.8A — Publication-quality weakly nonlinear bifurcation branch

Cooperative Fisher-KPP-chemotaxis system

Generated figures
-----------------
figure4_bifurcation_branch.png
figure4_bifurcation_branch.pdf

Required input
--------------
results_task17B_advanced/
    branch_continuation_advanced.csv
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

OUT = Path("figures_task18")
OUT.mkdir(exist_ok=True)

BRANCH_FILE = Path(
    "results_task17B_advanced/"
    "branch_continuation_advanced.csv"
)


# ============================================================
# Model / bifurcation parameters
# ============================================================

chi2 = 2.1143925003845543


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
# ============================================================

BLUE = "#2166AC"
ORANGE = "#D95F02"
DARK = "#222222"
GREY = "#737373"
LIGHT_GREY = "#D9D9D9"


# ============================================================
# Utility functions
# ============================================================

def read_csv(path):
    """
    Read CSV file as a list of dictionaries.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path.resolve()}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_figure(fig, basename):
    """
    Save publication-quality PNG and PDF versions.
    """

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


# ============================================================
# Branch data
# ============================================================

def get_branch_data():
    """
    Load numerical continuation data.
    """

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


# ============================================================
# Weakly nonlinear branch plot
# ============================================================

def plot_branch(ax):

    dchi, A = get_branch_data()

    # numerical squared amplitude
    A2 = A ** 2

    # leading-order theory evaluated at data points
    theory_at_data = (
        dchi
        /
        chi2
    )

    # smooth theoretical branch
    xx = np.linspace(
        0.0,
        dchi.max() * 1.10,
        500
    )

    theory = (
        xx
        /
        chi2
    )

    # --------------------------------------------------------
    # Leading-order theoretical branch
    # --------------------------------------------------------

    ax.plot(
        xx,
        theory,
        color=DARK,
        lw=1.7,
        zorder=2
    )

    # --------------------------------------------------------
    # Numerical continuation branch
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

    # --------------------------------------------------------
    # Threshold point
    # --------------------------------------------------------

    ax.scatter(
        [0.0],
        [0.0],
        s=33,
        facecolor=ORANGE,
        edgecolor="white",
        linewidth=0.6,
        zorder=6
    )

    # --------------------------------------------------------
    # Direct labels
    #
    # Main adjustment:
    # move the theory label to the left so that it does not
    # overlap the theoretical and numerical branches.
    # --------------------------------------------------------

    ax.text(
        0.00465,
        0.00318,
        r"theory:"
        "\n"
        r"$A^2=(\chi-\chi_T)/\chi_2$",
        color=DARK,
        fontsize=9.5,
        ha="left",
        va="center"
    )

    ax.text(
        0.0018,
        0.00165,
        "numerical branch",
        color=BLUE,
        fontsize=9.6,
        ha="left",
        va="center"
    )

    ax.text(
        0.00035,
        0.00010,
        r"$\chi=\chi_T$",
        fontsize=9.1,
        color=ORANGE,
        ha="left",
        va="center"
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
        0.0,
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

    # restore all four inset spines
    inset.spines["top"].set_visible(True)
    inset.spines["right"].set_visible(True)
    inset.spines["left"].set_visible(True)
    inset.spines["bottom"].set_visible(True)

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


# ============================================================
# Figure wrapper
# ============================================================

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
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Task 1.8A — weakly nonlinear bifurcation branch")
    print("=" * 70)

    figure_branch()

    print()
    print("=" * 70)
    print("Figure generation finished.")
    print("Saved in:")
    print(OUT.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()