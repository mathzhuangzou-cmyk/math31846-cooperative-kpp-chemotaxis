#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 1.9A — Real dispersion curve generation

Replace the previous schematic Figure 1 by the exact model-based curve:

chi(mu)=
(Dw*mu+alpha)*((d1*mu+a)(d2*mu+b)-k12*k21)
/
(beta*mu*(d2*mu+b))

Output:
    figures_task19/figure1_dispersion_real.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parameters from Task 1.6B
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
        (Dw*mu + alpha)
        *
        (
            (d1*mu+a)*(d2*mu+b)
            -
            k12*k21
        )
    )

    denominator = (
        beta*mu*(d2*mu+b)
    )

    return numerator/denominator



# ============================================================
# Generate figure
# ============================================================

def main():

    out = Path("figures_task19")
    out.mkdir(exist_ok=True)


    mu = np.linspace(
        0.05,
        8.0,
        1000
    )


    chi = chi_of_mu(mu)


    plt.figure(
        figsize=(6.5,4.5)
    )


    plt.plot(
        mu,
        chi,
        label=r"$\chi(\mu)$"
    )


    plt.scatter(
        [mu_c],
        [chi_c],
        label=r"$(\mu_c,\chi_c)$"
    )


    # Neumann modes
    n_values = np.arange(
        1,
        9
    )

    mu_n = (
        n_values*np.pi/L
    )**2

    chi_n = chi_of_mu(mu_n)


    plt.scatter(
        mu_n,
        chi_n,
        label=r"Neumann modes"
    )


    for n, x, y in zip(
        n_values,
        mu_n,
        chi_n
    ):

        if n in [5]:
            plt.annotate(
                f"$n={n}$",
                (x,y),
                xytext=(5,5),
                textcoords="offset points"
            )


    plt.xlabel(
        r"$\mu$"
    )

    plt.ylabel(
        r"$\chi(\mu)$"
    )


    plt.title(
        "Exact dispersion relation and discrete modes"
    )


    plt.legend()

    plt.grid(True)

    plt.tight_layout()


    plt.savefig(
        out/"figure1_dispersion_real.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Saved:",
        (out/"figure1_dispersion_real.png").resolve()
    )


if __name__=="__main__":
    main()
