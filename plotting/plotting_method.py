import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cvxpy as cp
import torch
from model_def_and_weights.model_weights import construct_problem_data_fixed, create_A_matrix

class SupportLP_ICNN:
    def __init__(self, model, solver="ECOS", verbose=False):
        self.model = model
        self.solver = solver
        self.verbose = verbose

    def argmax_u(self, w):
        C, d = construct_problem_data_fixed(self.model, [])
        A = create_A_matrix(self.model.input_dim, C.shape[1])
        d = np.asarray(d).reshape(-1)
        n = A.shape[1] 
        
        x = cp.Variable(n)
        w = np.asarray(w).reshape(-1)
        
        obj = cp.Maximize(w @ (A @ x))
        cons = [C @ x <= d]
        
        prob = cp.Problem(obj, cons)
        prob.solve(solver=self.solver, verbose=self.verbose)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"ICNN LP status: {prob.status}")
        x_star = x.value
        return A @ x_star


class SupportLP_PolytopeU:
    def __init__(self, H, h, solver="ECOS", verbose=False):
        self.H = np.asarray(H)
        self.h = np.asarray(h).reshape(-1)
        self.solver = solver
        self.verbose = verbose

        m = self.H.shape[1]
        self.u = cp.Variable(m)
        self.w = cp.Parameter(m)

        obj = cp.Maximize(self.w @ self.u)
        cons = [self.H @ self.u <= self.h]
        self.prob = cp.Problem(obj, cons)

    def argmax_u(self, w):
        w = np.asarray(w).reshape(-1)
        self.w.value = w
        self.prob.solve(solver=self.solver, warm_start=True, verbose=self.verbose)
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Polytope LP status: {self.prob.status}")
        return self.u.value


class Support_MinkowskiSum:
    def __init__(self, H, h_list, solver="ECOS", verbose=False):
        self.summands = [
            SupportLP_PolytopeU(H, h, solver=solver, verbose=verbose)
            for h in h_list
        ]

    def argmax_u(self, w):
        u_sum = None
        for lp in self.summands:
            u_i = lp.argmax_u(w)
            u_sum = u_i if u_sum is None else (u_sum + u_i)
        return u_sum

def _to_numpy(x):
    if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    return np.asarray(x)

def make_orthonormal_basis(v1):
    v1 = v1.reshape(-1)
    n = v1.shape[0]
    v1 = -v1 / v1.norm()

    A = torch.randn(n, n, device=v1.device, dtype=v1.dtype)
    A[:, 0] = v1
    Q, _ = torch.linalg.qr(A)
    return Q

def boundary_on_slice_via_support(argmax_u, u0, v1, vj, K=360):
    u0 = _to_numpy(u0).reshape(-1)
    v1 = _to_numpy(v1).reshape(-1); v1 = v1 / np.linalg.norm(v1)
    vj = _to_numpy(vj).reshape(-1); vj = vj / np.linalg.norm(vj)

    a_list, b_list, u_list = [], [], []
    for k in range(K):
        theta = 2.0 * math.pi * k / K
        w = math.cos(theta) * v1 + math.sin(theta) * vj
        u_star = argmax_u(w)
        u_star = np.asarray(u_star).reshape(-1)

        du = u_star - u0
        a_list.append(float(v1 @ du))
        b_list.append(float(vj @ du))
        u_list.append(u_star)

    return np.array(a_list), np.array(b_list)


def project_points_onto_slice(points, u0, v1, vj):
    P = _to_numpy(points)
    u0 = _to_numpy(u0).reshape(-1)
    v1 = _to_numpy(v1).reshape(-1); v1 = v1 / np.linalg.norm(v1)
    vj = _to_numpy(vj).reshape(-1); vj = vj / np.linalg.norm(vj)

    D = P - u0[None, :]
    a = D @ v1
    b = D @ vj
    return a, b

def plot_icnn_slices(
    u0,
    model,
    H,
    h_i_list, 
    title="2D Slice Boundary",
    ga_model = None, 
    sp_model = None, 
    opt_sols = None, 
    K=360,
    figsize=(10, 10),
):
    basis = _to_numpy(make_orthonormal_basis(u0))

    m = basis.shape[0]
    v1 = basis[:, 0]
    figs = []
    
    icnn_oracle = SupportLP_ICNN(model)
    exact_oracle = Support_MinkowskiSum(H, h_i_list)
    ga_oracle = SupportLP_PolytopeU(ga_model[0], ga_model[1]) if ga_model is not None else None
    sp_oracle = SupportLP_PolytopeU(sp_model[0], sp_model[1]) if sp_model is not None else None

    for idx in range(1, m):
        vj = basis[:, idx]
        
        a_icnn, b_icnn = boundary_on_slice_via_support(icnn_oracle.argmax_u, u0, v1, vj, K=K)
        a_ex, b_ex = boundary_on_slice_via_support(exact_oracle.argmax_u, u0, v1, vj, K=K)
        if ga_oracle is not None:
            a_ga, b_ga = boundary_on_slice_via_support(ga_oracle.argmax_u, u0, v1, vj, K=K)
        if sp_oracle is not None:
            a_sp, b_sp = boundary_on_slice_via_support(sp_oracle.argmax_u, u0, v1, vj, K=K)
        
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        
        if opt_sols is not None:
            a_pts, b_pts = project_points_onto_slice(opt_sols, u0, v1, vj)
            ax.scatter(a_pts, b_pts, s=25, alpha=0.9, label="Other points")

        ax.plot(np.r_[a_icnn, a_icnn[0]], np.r_[b_icnn, b_icnn[0]],
                lw=2, color="purple")
        if ga_oracle is not None:
            ax.plot(np.r_[a_ga, a_ga[0]], np.r_[b_ga, b_ga[0]],
            lw=2, color="green", ls="--")
        if sp_oracle is not None:
            ax.plot(np.r_[a_sp, a_sp[0]], np.r_[b_sp, b_sp[0]],
            lw=2, color="red", ls = "--")
        ax.plot(np.r_[a_ex, a_ex[0]], np.r_[b_ex, b_ex[0]],
                lw=1.5, ls="--", color="black")

        ax.set_xlabel("Along v1")
        ax.set_ylabel(f"Along v{idx+1}")
        ax.set_title(f"{title}")

        # Legend
        line_list = []
        line1 = Line2D([], [], linewidth=2, color="purple", label="ICNN")
        line_list.append(line1)
        if ga_oracle is not None:
            line2 = Line2D([], [], linestyle="--", color="green", linewidth=1.5, label="Gen. Affine")
            line_list.append(line2)
        if sp_oracle is not None:
            line3 = Line2D([], [], linestyle="--", color="red", linewidth=1.5, label="Struct. Preserving")
            line_list.append(line3)
        line4 = Line2D([], [], linestyle="--", color="black", linewidth=1.5, label="Minkowski Sum")
        line_list.append(line4)
        ax.legend(handles=line_list, loc="upper right")

        figs.append(fig)
        plt.close(fig)

    return figs
