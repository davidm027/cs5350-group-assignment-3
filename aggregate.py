"""
How to use this file:
1. Make sure you have the Pandas library installed
2. Run `aggregate()` in root directory
3. Check `results/final` directory for the files containing all the data for each experiment
4. When plotting: get a dataframe by calling `load_histories()` on the file in `results/final` you want to plot, then use matplotlib to plot
"""

import os
from pathlib import Path

import pandas as pd


def line_prepender(filename, line):
    with open(filename, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip("\r\n") + "\n" + content)


def load_histories(file, pattern):
    # Pattern for matrix dimension experiments: r"^m\d{3,4}_n\d{3,4}_q\d{3,4}_P\d{2,3}_seed(\d)\.txt$"
    # Pattern for processor count experiments: r"^dims\d{3,4}_P\d{1,2}_seed(\d)\.txt$"
    line_prepender(file, "m,n,q,P,MM_ser_sec,MM_Par_sec,MM_1D_sec,MM_2D_sec,seed")

    out = pd.read_csv(file)

    # Basic cleanup
    out.columns = [c.strip() for c in out.columns]
    out["m"] = out["m"].astype(int)
    out["n"] = out["n"].astype(int)
    out["q"] = out["q"].astype(int)
    out["P"] = out["P"].astype(int)
    out["MM_ser_sec"] = out["MM_ser_sec"].astype(float)
    out["MM_Par_sec"] = out["MM_Par_sec"].astype(float)
    out["MM_1D_sec"] = out["MM_1D_sec"].astype(float)
    out["MM_2D_sec"] = out["MM_2D_sec"].astype(float)

    out = out.groupby(["experiment", "m", "n", "q", "P"], as_index=False)[
        ["MM_ser_s", "MM_par_s", "MM_1d_s", "MM_2d_s"]
    ].mean()

    return out


def aggregate():
    os.makedirs("results/final/", exist_ok=True)
    for experiment in [dir for dir in os.listdir(f"results/") if "experiment" in dir]:
        filenames = os.listdir(f"results/{experiment}")
        with open(f"results/final/{experiment}-total.csv", "w") as new_file:
            for name in filenames:
                with open(f"results/{experiment}/{name}") as f:
                    for line in f:
                        new_file.write(line)
