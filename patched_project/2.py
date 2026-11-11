#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 1.7 — Numerical Audit

Compatible with user's Task 1.6B file: 1.py

Validation:
1. Time-step convergence
2. Branch prediction table
3. Spatial resolution audit

GPU compatible.
"""


from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Task 1.6B dynamic import
# ============================================================

TASK16_FILE = "1.py"


def load_task16_module():

    file_path = Path(__file__).parent / TASK16_FILE

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot find {TASK16_FILE}"
        )

    module_name = "task16_module"

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    module = importlib.util.module_from_spec(spec)

    # ================================
    # Important:
    # register module before execution
    # required by dataclass
    # ================================
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


task16 = load_task16_module()


ModelParams = task16.ModelParams
SpectralIMEX1D = task16.SpectralIMEX1D
continuous_critical_data = task16.continuous_critical_data
select_mode = task16.select_mode
chi2_data = task16.chi2_data
dominant_linear_mode = task16.dominant_linear_mode



# ============================================================
# CSV writer
# ============================================================

def write_csv(path, rows):

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
# 1. Time convergence
# ============================================================

def time_convergence():

    print("\nRunning time convergence...")


    p = ModelParams()

    L = 10.0


    critical = continuous_critical_data(p)

    mode = select_mode(
        L,
        p,
        critical["mu_c"]
    )


    chi = 1.02 * mode["chi_T"]


    lam, _ = dominant_linear_mode(
        p,
        mode["mu_star"],
        chi
    )


    rows=[]


    for dt in [
        0.04,
        0.02,
        0.01,
        0.005
    ]:

        lam_imex = task16.actual_imex_growth_rate(
            p, mode["mu_star"], chi, dt
        )


        rows.append(
            {
                "dt":dt,
                "lambda_exact":lam,
                "lambda_imex":lam_imex,
                "absolute_error":
                    abs(lam-lam_imex)
            }
        )


        print(
            f"dt={dt:.5f}, "
            f"error={abs(lam-lam_imex):.6e}"
        )


    write_csv(
        "results_task17/time_convergence.csv",
        rows
    )



# ============================================================
# 2. Bifurcation prediction
# ============================================================

def branch_prediction():

    print("\nGenerating branch prediction table...")


    p = ModelParams()

    L = 10.0


    critical = continuous_critical_data(p)

    mode = select_mode(
        L,
        p,
        critical["mu_c"]
    )


    c2 = chi2_data(
        L,
        p,
        mode["n_star"],
        mode["mu_star"],
        mode["chi_T"]
    )


    rows=[]


    for delta in [
        0.00125,
        0.0025,
        0.005,
        0.01
    ]:


        A = np.sqrt(
            delta/c2["chi2"]
        )


        rows.append(
            {
                "delta_chi":delta,
                "chi2_theory":c2["chi2"],
                "A_prediction":A
            }
        )


        print(
            f"delta={delta:.6f}, "
            f"A={A:.8e}"
        )


    write_csv(
        "results_task17/branch_prediction.csv",
        rows
    )



# ============================================================
# 3. Spatial resolution
# ============================================================

def spatial_resolution(device):

    print("\nListing grid sizes only (no PDE spatial-convergence test)...")


    p = ModelParams()


    rows=[]


    for n_half in [
        129,
        257,
        513,
        1025
    ]:


        solver = SpectralIMEX1D(
            p,
            10.0,
            n_half,
            0.02,
            device,
            torch.float32
        )


        rows.append(
            {
                "n_half":n_half,
                "FFT_points":2*(n_half-1),
                "dx":10.0/(n_half-1)
            }
        )


        print(
            f"M={2*(n_half-1)}"
        )


    write_csv(
        "results_task17/spatial_resolution.csv",
        rows
    )



# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--device",
        default="cuda"
    )


    args = parser.parse_args()



    if (
        args.device=="cuda"
        and torch.cuda.is_available()
    ):

        device=torch.device("cuda")

    else:

        device=torch.device("cpu")



    print("="*70)
    print("Task 1.7 Numerical Audit")
    print("="*70)

    print(
        "Device:",
        device
    )


    if device.type=="cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    print("="*70)



    time_convergence()

    branch_prediction()

    spatial_resolution(device)



    print("="*70)
    print("Task 1.7 finished")
    print(
        "Results saved in:",
        Path("results_task17").resolve()
    )
    print("="*70)



if __name__=="__main__":

    main()