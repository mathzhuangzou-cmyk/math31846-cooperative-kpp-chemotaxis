#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 1.7B-Advanced
Nonlinear continuation validation with:
1. true critical eigenvector initialization
2. second-order correction terms
3. numerical verification of

        chi-chi_T = chi_2 A^2 + o(A^2)

Compatible with Task 1.6B solver (1.py).
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


TASK16_FILE = "1.py"


# ============================================================
# Load Task 1.6B
# ============================================================

def load_task16():

    path = Path(__file__).parent / TASK16_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {TASK16_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        "task16_module",
        path
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules["task16_module"] = module

    spec.loader.exec_module(module)

    return module


task16 = load_task16()

ModelParams = task16.ModelParams
SpectralIMEX1D = task16.SpectralIMEX1D
continuous_critical_data = task16.continuous_critical_data
select_mode = task16.select_mode
chi2_data = task16.chi2_data
dominant_linear_mode = task16.dominant_linear_mode


# ============================================================
# Utilities
# ============================================================

def save_csv(path, rows):

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Second-order corrected initialization
# ============================================================

def initialize_center_manifold_state(
        solver,
        p,
        n_star,
        A,
        chi,
        c2):

    x = solver.x

    mode1 = torch.cos(
        n_star*np.pi*x/solver.L
    )

    mode2 = torch.cos(
        2*n_star*np.pi*x/solver.L
    )


    # True critical eigenvector
    _, vec = dominant_linear_mode(
        p,
        (n_star*np.pi/solver.L)**2,
        chi
    )


    U1, V1, W1 = vec


    u = (
        1
        + A*U1*mode1
        + A*A*(c2["U0"] + c2["U2"]*mode2)
    )

    v = (
        1
        + A*V1*mode1
        + A*A*(c2["V0"] + c2["V2"]*mode2)
    )

    w = (
        p.beta/p.alpha
        + A*W1*mode1
        + A*A*(c2["W0"] + c2["W2"]*mode2)
    )

    return u,v,w



# ============================================================
# Advanced continuation
# ============================================================

def run():

    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("="*70)
    print("Task 1.7B Advanced Continuation")
    print("Device:",device)
    print("="*70)


    p=ModelParams()

    L=10.0


    critical=continuous_critical_data(p)

    mode=select_mode(
        L,
        p,
        critical["mu_c"]
    )


    chi_T=mode["chi_T"]

    chi2_info=chi2_data(
        L,
        p,
        mode["n_star"],
        mode["mu_star"],
        chi_T
    )

    chi2=chi2_info["chi2"]


    solver=SpectralIMEX1D(
        p,
        L,
        513,
        0.01,
        device,
        torch.float64
    )


    rows=[]


    for delta in [
        0.00125,
        0.0025,
        0.005,
        0.01
    ]:

        chi=chi_T+delta


        A0=np.sqrt(
            delta/chi2
        )


        u,v,w=initialize_center_manifold_state(
            solver,
            p,
            mode["n_star"],
            A0,
            chi,
            chi2_info
        )


        steps=int(500/solver.dt)


        for _ in range(steps):

            u,v,w=solver.step(
                u,
                v,
                w,
                chi
            )


        A_num=abs(
            solver.cosine_amplitude(
                u,
                mode["n_star"],
                1.0
            )
        )


        ratio=delta/(A_num*A_num)


        rows.append(
            {
                "delta_chi":delta,
                "A_initial":A0,
                "A_numeric":A_num,
                "delta_over_A2":ratio,
                "chi2_theory":chi2
            }
        )


        print(
            f"delta={delta:.6f}, "
            f"A={A_num:.8e}, "
            f"ratio={ratio:.8f}"
        )


    save_csv(
        "results_task17B_advanced/"
        "branch_continuation_advanced.csv",
        rows
    )


    print("-"*70)
    print("Theory chi2 =",chi2)
    print("Results saved.")
    print("="*70)



if __name__=="__main__":
    run()
