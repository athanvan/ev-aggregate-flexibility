import numpy as np
import pandas as pd
from scipy.linalg import block_diag


def f_reshape(a, newshape):
    return np.reshape(a, newshape, order="F")

def f_vec(a):
    return np.reshape(a, (-1,), order="F")

def blkdiag_repeat(M, N):
    mats = [M] * N
    return block_diag(*mats) if mats else np.zeros((0, 0))

def load_household_15min(path_csv: str) -> pd.DataFrame:
    """
    Args:
        path_csv: path to CSV containing 15-minute load data for New York households,
            CSV should have columns ['dataid', 'local_15min', 'grid', 'solar', 'solar2']

    Returns:
        DataFrame with the following columns, sorted in chronological order
        (by 'local_15min'):
            - 'dataid': household id
            - 'local_15min': timestamp (15-min resolution)
            - 'grid': load from grid
            - 'solar': load from solar panel 1
            - 'solar2': load from solar panel 2
            - 'use': total load (grid + solar + solar2)
    """
    nys = pd.read_csv(
        path_csv, usecols=["dataid", "local_15min", "grid", "solar", "solar2"])
    nys = nys.sort_values("local_15min")

    # NaNs to 0 for solar / solar2
    nys[["solar", "solar2"]] = nys[["solar", "solar2"]].fillna(0.)

    nys["use"] = nys["grid"].to_numpy() + nys["solar"].to_numpy() + nys["solar2"].to_numpy()
    return nys

def build_load_profiles(nys: pd.DataFrame, T_15: int = 24*4, expected_days: int = 184) -> np.ndarray:
    """
    Args:
        nys: dataframe with columns ['dataid', 'local_15min', 'grid', 'solar', 'solar2', 'use']
        T_15: number of 15-minute intervals per day
        expected_days: number of days to consider

    Returns:
        agg_load
    """
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
