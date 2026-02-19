"""Utilities for plotting 2D slices of flexible-set representations."""

import math
from collections.abc import Callable, Sequence
from typing import Any

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from model_def_and_weights.model_weights import (
    construct_problem_data_fixed,
    create_A_matrix,
)


def _to_numpy(x: Any) -> npt.NDArray[Any]:
    """Convert tensor-like inputs to NumPy arrays."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Slice_ICNN:
    """Oracle for ray-casting against the ICNN feasible set."""

    def __init__(self, model, solver, verbose: bool = False) -> None:
        """Store model and solver settings used by the CVXPY problem."""
        self.model = model
        self.solver = solver
        self.verbose = verbose

    def max_radius(
        self,
        u0: npt.ArrayLike,
        d: npt.ArrayLike,
    ) -> float:
        """Calculates the largest feasible radius from ``u0`` along direction ``d``.

        Args:
            u_0: origin of slice, array of shape [input_dim]
            d: direction of slice, array of shape [input_dim]
        """
        C, d_vec = construct_problem_data_fixed(self.model, [])
        A = create_A_matrix(self.model.input_dim, C.shape[1])
        d_vec = np.asarray(d_vec).reshape(-1)
        n = A.shape[1]

        x = cp.Variable(n)
        r = cp.Variable(nonneg=True)

        obj = cp.Maximize(r)
        cons = [C @ x <= d_vec, A @ x == u0 + r * d]

        prob = cp.Problem(obj, cons)
        prob.solve(solver=self.solver, verbose=self.verbose)

        if prob.status in ("optimal", "optimal_inaccurate"):
            assert r.value is not None
            return float(r.value)
        return 0.0


class Slice_PolytopeU:
    """Oracle for ray-casting against a polytope in the ``u`` variables."""

    def __init__(
        self,
        H: npt.ArrayLike,
        h: npt.ArrayLike,
        solver,
        verbose: bool = False,
    ) -> None:
        """Construct the reusable CVXPY model for polytope queries."""
        self.H = np.asarray(H)
        self.h = np.asarray(h).reshape(-1)
        self.solver = solver
        self.verbose = verbose

        self.m = self.H.shape[1]
        self.r = cp.Variable(nonneg=True)
        self.u0_param = cp.Parameter(self.m)
        self.d_param = cp.Parameter(self.m)

        obj = cp.Maximize(self.r)
        cons = [self.H @ (self.u0_param + self.r * self.d_param) <= self.h]
        self.prob = cp.Problem(obj, cons)

    def max_radius(
        self,
        u0: npt.ArrayLike,
        d: npt.ArrayLike,
    ) -> float:
        """Return the largest feasible radius from ``u0`` along direction ``d``."""
        self.u0_param.value = u0
        self.d_param.value = d
        self.prob.solve(
            solver=self.solver,
            warm_start=True,
            verbose=self.verbose,
        )
        if self.prob.status in ("optimal", "optimal_inaccurate"):
            assert self.r.value is not None
            return float(self.r.value)
        return 0.0


class Slice_MinkowskiSum:
    """Oracle for the Minkowski sum represented by shared halfspaces."""

    def __init__(
        self,
        H: npt.ArrayLike,
        h_list: Sequence[npt.ArrayLike],
        solver,
        verbose: bool = False,
    ) -> None:
        """Construct the reusable CVXPY model for Minkowski-sum queries."""
        self.H = np.asarray(H)
        self.h_list = [np.asarray(h).reshape(-1) for h in h_list]
        self.solver = solver
        self.verbose = verbose

        self.N = len(self.h_list)
        self.m = self.H.shape[1]

        self.U = cp.Variable((self.N, self.m))
        self.r = cp.Variable(nonneg=True)

        self.u0_param = cp.Parameter(self.m)
        self.d_param = cp.Parameter(self.m)

        obj = cp.Maximize(self.r)
        cons = [
            self.H @ self.U[i] <= self.h_list[i]
            for i in range(self.N)
        ]
        cons.append(
            cp.sum(self.U, axis=0) == self.u0_param + self.r * self.d_param
        )
        self.prob = cp.Problem(obj, cons)

    def max_radius(
        self,
        u0: npt.ArrayLike,
        d: npt.ArrayLike,
    ) -> float:
        """Return the largest feasible radius from ``u0`` along direction ``d``."""
        self.u0_param.value = u0
        self.d_param.value = d
        self.prob.solve(
            solver=self.solver,
            warm_start=True,
            verbose=self.verbose,
        )
        if self.prob.status in ("optimal", "optimal_inaccurate"):
            assert self.r.value is not None
            return float(self.r.value)
        return 0.0


def make_custom_basis(
    u0: npt.ArrayLike,
    u_icnn: npt.ArrayLike,
    u_taha: npt.ArrayLike,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Build an orthonormal basis from the ICNN and TAHA directions.

    Args:
        u0: u_optimal, array of shape [T]
        u_icnn: ICNN solution, array of shape [T]
        u_taha: TAHA solution, array of shape [T]

    Returns:
        v1: unit vector in the direction of u_icnn - u0, array of shape [T]
        v2: unit vector, projection of u_taha - u0 orthogonal to v1,
            array of shape [T]
    """
    u0 = _to_numpy(u0).reshape(-1)
    u_icnn = _to_numpy(u_icnn).reshape(-1)
    u_taha = _to_numpy(u_taha).reshape(-1)

    v1 = u_icnn - u0
    v1 /= np.linalg.norm(v1)

    v2 = u_taha - u0
    v2 /= np.linalg.norm(v2)

    new_v2 = v2 - (v1 @ v2) * v1
    new_v2 /= np.linalg.norm(new_v2)

    return v1, new_v2


def boundary_on_slice_via_raycast(
    max_radius_fn: Callable[[npt.ArrayLike, npt.ArrayLike], float],
    u0: npt.ArrayLike,
    v1: npt.ArrayLike,
    v2: npt.ArrayLike,
    a: float,
    b: float,
    K: int = 360,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Sample a 2D slice boundary by ray-casting from a slice origin.

    Args:
        max_radius_fn: function that takes in (u0, d) and returns the maximum feasible
            radius along direction d from u0
        u0: slice origin, array of shape [T]
        v1: first basis, unit vector, array of shape [T]
        v2: second basis, unit vector orthogonal to v1, array of shape [T]
        a: coordinate of slice origin along v1, scalar
        b: coordinate of slice origin along v2, scalar
        K: number of rays to cast, i.e. number of points to sample on the boundary
    """
    u0 = _to_numpy(u0).reshape(-1)
    v1 = _to_numpy(v1).reshape(-1)
    v2 = _to_numpy(v2).reshape(-1)

    a_list, b_list = [], []
    for k in range(K):
        theta = 2.0 * math.pi * k / K
        d = math.cos(theta) * v1 + math.sin(theta) * v2

        r_max = max_radius_fn(u0, d)

        a_list.append(a + r_max * math.cos(theta))
        b_list.append(b + r_max * math.sin(theta))

    return np.array(a_list), np.array(b_list)


def project_point_onto_basis(
    point: npt.ArrayLike,
    u0: npt.ArrayLike,
    v1: npt.ArrayLike,
    v2: npt.ArrayLike,
) -> tuple[float, float]:
    """Project a point onto the 2D slice basis coordinates.

    Args:
        point: point to project, array of shape [T]
        u0: slice origin, array of shape [T]
        v1: first basis, array of shape [T]
        v2: second basis, orthogonal to v1, array of shape [T]

    Returns:
        a: coordinate of point (relative to u_0) along v1, scalar
        b: coordinate of point (relative to u_0) along v2, scalar
    """
    point = _to_numpy(point).reshape(-1)
    u0 = _to_numpy(u0).reshape(-1)
    v1 = _to_numpy(v1).reshape(-1)
    v2 = _to_numpy(v2).reshape(-1)

    # normalize in case not already unit vectors
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    D = point - u0
    a = D @ v1
    b = D @ v2
    return float(a), float(b)


def plot_specific_slice(
    u0: npt.ArrayLike,
    u_icnn: npt.ArrayLike,
    u_taha: npt.ArrayLike,
    model,
    H: npt.ArrayLike,
    h_i_list: Sequence[npt.ArrayLike],
    ga_model=None,
    K: int = 360,
    figsize: tuple[float, float] = (5, 5),
    solver="CUOPT",
) -> Figure:
    """Plot the ICNN, exact, and optional affine slice boundaries."""
    v1, v2 = make_custom_basis(u0, u_icnn, u_taha)
    a_i, b_i = project_point_onto_basis(u_icnn, u0, v1, v2)
    a_t, b_t = project_point_onto_basis(u_taha, u0, v1, v2)

    icnn_oracle = Slice_ICNN(model, solver)
    exact_oracle = Slice_MinkowskiSum(H, h_i_list, solver)
    ga_oracle = (
        Slice_PolytopeU(ga_model[0], ga_model[1], solver)
        if ga_model is not None
        else None
    )

    a_icnn, b_icnn = boundary_on_slice_via_raycast(
        icnn_oracle.max_radius,
        u_icnn,
        v1,
        v2,
        a_i,
        b_i,
        K=K,
    )
    a_ex, b_ex = boundary_on_slice_via_raycast(
        exact_oracle.max_radius,
        u0,
        v1,
        v2,
        0,
        0,
        K=K,
    )

    if ga_oracle is not None:
        a_ga, b_ga = boundary_on_slice_via_raycast(
            ga_oracle.max_radius,
            u_taha,
            v1,
            v2,
            a_t,
            b_t,
            K=K,
        )

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    ax.scatter(
        0, 0,
        marker="*", s=150, color="black", zorder=5, edgecolors="black", label="u_OPTIMAL"
    )
    ax.scatter(
        a_i, b_i,
        marker="P", s=100, color="tab:blue", zorder=5, label="u_ICNN"
    )
    ax.scatter(
        a_t, b_t,
        marker="P", s=100, color="tab:orange", zorder=5, label="u_TAHA"
    )

    ax.plot(
        np.r_[a_icnn, a_icnn[0]],
        np.r_[b_icnn, b_icnn[0]],
        lw=2,
        ls="--",
        color="tab:blue",
        label="ICNN"
    )
    ax.plot(
        np.r_[a_ex, a_ex[0]],
        np.r_[b_ex, b_ex[0]],
        lw=1.5,
        ls="--",
        color="black",
        label="True Minkowski Sum"
    )

    if ga_oracle is not None:
        ax.plot(
            np.r_[a_ga, a_ga[0]],
            np.r_[b_ga, b_ga[0]],
            lw=2,
            color="tab:orange",
            ls="--",
            label="Gen. Affine"
        )
    ax.legend(loc="upper right")

    return fig
