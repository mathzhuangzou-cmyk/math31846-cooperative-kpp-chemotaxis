#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 1.8B - Final paper table generator

Generate manuscript tables from Task 1.7 results.
"""

from pathlib import Path
import csv


OUT = Path("tables_task18")
OUT.mkdir(exist_ok=True)


def write_table(name, header, rows):

    with open(
        OUT/name,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():

    write_table(
        "critical_parameters.csv",
        ["parameter","value"],
        [
            ["mu_c","2.468635740034"],
            ["k_c","1.571189275687"],
            ["chi_c","2.452720209264"],
            ["Lambda_c","3.998999614119"],
            ["n_star","5"],
            ["chi_T(L)","2.452720334543"]
        ]
    )


    write_table(
        "branch_validation.csv",
        [
            "delta_chi",
            "A_numeric",
            "delta_over_A2",
            "chi2_theory"
        ],
        [
            [
                "0.00125",
                "0.024304443",
                "2.11611209",
                "2.114392500"
            ],
            [
                "0.00250",
                "0.034344667",
                "2.11944128",
                "2.114392500"
            ],
            [
                "0.00500",
                "0.048480668",
                "2.12731997",
                "2.114392500"
            ],
            [
                "0.01000",
                "0.068316579",
                "2.14263292",
                "2.114392500"
            ]
        ]
    )


    print("Tables generated:", OUT.resolve())


if __name__=="__main__":
    main()
