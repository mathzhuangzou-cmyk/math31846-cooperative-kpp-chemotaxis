#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 1.9A — Publication quality dispersion figure

Exact dispersion relation:
chi(mu)=
(Dw*mu+alpha)
*((d1*mu+a)(d2*mu+b)-k12*k21)
/
(beta*mu*(d2*mu+b))

Output:
    figure1_dispersion_real_high.png
"""


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ============================================================
# Parameters
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


mu_c = 2.468635740034
chi_c = 2.452720209264

L = 10.0


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
        beta * mu * (d2 * mu + b)
    )

    return numerator / denominator


# ============================================================
# Figure
# ============================================================

def main():

    out = Path("figures_task19")
    out.mkdir(exist_ok=True)

    plt.rcParams.update({

        "font.family": "serif",

        "font.size": 12,

        "axes.labelsize": 14,

        "axes.titlesize": 14,

        "legend.fontsize": 10,

        "mathtext.fontset": "stix"

    })

    fig, ax = plt.subplots(

        figsize=(7.2, 4.8),

        dpi=300

    )

    # --------------------------------------------------------
    # Continuous curve
    # --------------------------------------------------------

    mu = np.linspace(

        0.05,

        8,

        2000

    )

    chi = chi_of_mu(mu)

    ax.plot(

        mu,

        chi,

        linewidth=2.5,

        label=r"$\chi(\mu)$"

    )

    # subtle fill

    ax.fill_between(

        mu,

        chi,

        chi.min(),

        alpha=0.12

    )

    # --------------------------------------------------------
    # Critical point
    # --------------------------------------------------------

    ax.scatter(

        mu_c,

        chi_c,

        s=90,

        zorder=5,

        label=r"$(\mu_c,\chi_c)$"

    )

    ax.annotate(

        r"$\mu_c=2.4686$" + "\n" +
        r"$\chi_c=2.4527$",

        xy=(mu_c, chi_c),

        xytext=(3.5, 6.15),

        arrowprops=dict(

            arrowstyle="->",

            lw=1.2

        )

    )

    # --------------------------------------------------------
    # Neumann spectrum
    # --------------------------------------------------------

    n_values = np.arange(1, 9)

    mu_n = (n_values * np.pi / L) ** 2

    chi_n = chi_of_mu(mu_n)

    ax.scatter(

        mu_n,

        chi_n,

        s=65,

        marker="o",

        zorder=6,

        label="Neumann modes"

    )

    # highlight n=5

    idx = 4

    ax.scatter(

        mu_n[idx],

        chi_n[idx],

        s=140,

        facecolors="none",

        linewidths=2.0,

        zorder=7

    )

    ax.annotate(

        r"$n_*=5$",

        xy=(mu_n[idx], chi_n[idx]),

        xytext=(

            mu_n[idx] + 0.7,

            chi_n[idx] + 1.65

        ),

        arrowprops=dict(

            arrowstyle="->",

            lw=1.1

        )

    )

    # --------------------------------------------------------
    # inset zoom
    # --------------------------------------------------------

    # Shift the entire inset further downward
    # to avoid overlap with the legend and main title.

    axins = inset_axes(

        ax,

        width="38%",

        height="38%",

        loc="upper right",

        bbox_to_anchor=(0, -0.20, 1, 1),

        bbox_transform=ax.transAxes,

        borderpad=0.8

    )

    zoom_mu = np.linspace(

        1.8,

        3.2,

        400

    )

    axins.plot(

        zoom_mu,

        chi_of_mu(zoom_mu),

        linewidth=2

    )

    axins.scatter(

        mu_c,

        chi_c,

        s=40

    )

    axins.scatter(

        mu_n[idx],

        chi_n[idx],

        s=45

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

        "critical mode",

        fontsize=9

    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(

        r"$\mu$"

    )

    ax.set_ylabel(

        r"$\chi(\mu)$"

    )

    ax.set_title(

        "Exact dispersion relation and finite-domain mode selection"

    )

    ax.legend(

        frameon=False,

        loc="upper right"

    )

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.grid(

        alpha=0.25

    )

    plt.tight_layout()

    fig.savefig(

        out / "figure1_dispersion_real_high.png",

        dpi=600,

        bbox_inches="tight"

    )

    plt.close()

    print(

        "Saved:",

        (out / "figure1_dispersion_real_high.png").resolve()

    )


if __name__ == "__main__":

    main()