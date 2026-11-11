#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 1.7B — GPU nonlinear continuation validation

Goal:
Compute numerical stationary branches near chi_T and verify

        chi - chi_T = chi_2 A^2 + o(A^2)

against the theoretical coefficient chi_2.

This code uses the verified Task 1.6B solver (1.py).
"""

from __future__ import annotations

import importlib.util
import sys
import csv
from pathlib import Path
import numpy as np
import torch


TASK16_FILE = "1.py"


def load_task16():

    path = Path(__file__).parent / TASK16_FILE

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


def save_csv(path, rows):

    path = Path(path)
    path.parent.mkdir(
        exist_ok=True,
        parents=True
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



def initialize_state(
        solver,
        n_star,
        A):

    x = solver.x

    mode = torch.cos(
        n_star*np.pi*x/solver.L
    )

    u = 1.0 + A*mode
    v = 1.0 + A*mode
    w = 1.0 + A*mode

    return u,v,w



def continuation():

    device=torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    print("Device:",device)


    p=ModelParams()

    L=10.0


    critical=continuous_critical_data(p)

    mode=select_mode(
        L,
        p,
        critical["mu_c"]
    )


    c2=chi2_data(
        L,
        p,
        mode["n_star"],
        mode["mu_star"],
        mode["chi_T"]
    )


    chi_T=mode["chi_T"]

    chi2=c2["chi2"]


    solver=SpectralIMEX1D(
        p,
        L,
        513,
        0.02,
        device,
        torch.float32
    )


    rows=[]


    for dchi in [
        0.00125,
        0.0025,
        0.005,
        0.01
    ]:


        chi=chi_T+dchi


        A0=np.sqrt(
            dchi/chi2
        )


        u,v,w=initialize_state(
            solver,
            mode["n_star"],
            A0
        )


        steps=int(300/solver.dt)


        for _ in range(steps):

            u,v,w=solver.step(
                u,v,w,
                chi
            )


        A_num=abs(
            solver.cosine_amplitude(
                u,
                mode["n_star"],
                1.0
            )
        )


        rows.append(
            {
                "delta_chi":dchi,
                "A_theory":A0,
                "A_numeric":A_num,
                "delta_over_A2":
                    dchi/(A_num*A_num)
            }
        )


        print(
            dchi,
            A_num,
            dchi/(A_num*A_num)
        )


    save_csv(
        "results_task17B/branch_continuation.csv",
        rows
    )


    print(
        "Theory chi2:",
        chi2
    )


if __name__=="__main__":
    continuation()
