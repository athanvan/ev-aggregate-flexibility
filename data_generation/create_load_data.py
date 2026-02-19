import time
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from model_def_and_weights.icnn_definition import ICNN
import torch 

def f_reshape(a, newshape):
    return np.reshape(a, newshape, order="F")

def f_vec(a):
    return np.reshape(a, (-1,), order="F")

def blkdiag_repeat(M, N):
    mats = [M] * N
    return block_diag(*mats) if mats else np.zeros((0, 0))

def load_household_15min(path_csv):
    nys = pd.read_csv(path_csv)
    nys = nys[["dataid", "local_15min", "grid", "solar", "solar2"]].copy()
    nys = nys.sort_values("local_15min")

    # NaNs to 0 for solar / solar2
    for c in ["solar", "solar2"]:
        if c in nys.columns:
            nys[c] = nys[c].fillna(0.0)
        else:
            nys[c] = 0.0

    nys["use"] = nys["grid"].to_numpy() + nys["solar"].to_numpy() + nys["solar2"].to_numpy()
    return nys

def build_load_profiles(nys, T_15=24*4, expected_days=184):
    dataids = np.sort(nys["dataid"].unique())
    num_ids = len(dataids)

    load_profiles = np.zeros((T_15, num_ids, expected_days), dtype=float)

    i = 0
    for id_ in dataids:
        series = nys.loc[nys["dataid"] == id_, "use"].to_numpy()

        # Fix the missing entry for dataid==27
        if id_ == 27:
            if len(series) >= 8594:
                insert_val = series[8592:8594].mean()
                series = np.concatenate([series[:8593], np.array([insert_val]), series[8593:]])
            else:
                pass

        # Reshape (T_15, -1) to get days on the 2nd axis
        reshaped = f_reshape(series, (T_15, -1))
        if reshaped.shape[1] != expected_days:
            expected_days = reshaped.shape[1]
            # Resize container if first id; otherwise, ensure consistent shape
            if i == 0:
                load_profiles = np.zeros((T_15, num_ids, expected_days), dtype=float)

        load_profiles[:, i, :] = reshaped
        i += 1

    # sum across households
    agg_load = np.sum(load_profiles, axis=1)
    return agg_load
