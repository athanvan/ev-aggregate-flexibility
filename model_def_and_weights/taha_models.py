import numpy as np
import cvxpy as cp

def struct_preserve_inner_approx(H, h_i, h0, N):
    """
    From Taha et. al 2024, solving eq. 24 
    """
    alpha_in = cp.Variable(nonneg=True)
    r, c = H.shape
    pbar = cp.Variable(c)
    
    Gamma_i = [cp.Variable((c, c)) for _ in range(N)]
    gamma_i = [cp.Variable(c) for _ in range(N)]
    Lambda_i = [cp.Variable((r, r), nonneg=True) for _ in range(N)]
    gamma_sum = 0 
    Gamma_sum = 0
    I = np.eye(c)
    constraints = []
    
    for idx in range(N): 
        constraints.extend([
            Lambda_i[idx] @ H == H @ Gamma_i[idx],
            Lambda_i[idx] @ h0 <= h_i[:, idx] - H @ gamma_i[idx]
            ])
        gamma_sum += gamma_i[idx]
        Gamma_sum += Gamma_i[idx]

    constraints.append(pbar == gamma_sum)
    constraints.append(alpha_in * I == Gamma_sum)
    
    prob = cp.Problem(cp.Maximize(alpha_in), constraints)
    prob.solve(solver="CUOPT", verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("struct_preserve_inner_approx status:", prob.status)

    return float(alpha_in.value), pbar.value

def general_affine_inner_approx(H, h_i, h0, N):
    """
    From Taha et. al 2024, solving eq. 27
    """
    r, c = H.shape
    P = cp.Variable((c, c))
    n = len(h_i)
    pbar = cp.Variable(c)
    
    Gamma_i = [cp.Variable((c, c)) for _ in range(N)]
    gamma_i = [cp.Variable(c) for _ in range(N)]
    Lambda_i = [cp.Variable((r, r), nonneg=True) for _ in range(N)]
    gamma_sum = 0 
    Gamma_sum = 0
    constraints = []
    
    for idx in range(N): 
        constraints.extend([
            Lambda_i[idx] @ H == H @ Gamma_i[idx],
            Lambda_i[idx] @ h0 <= h_i[:, idx] - H @ gamma_i[idx]
            ])
        gamma_sum += gamma_i[idx]
        Gamma_sum += Gamma_i[idx]

    constraints.append(pbar == gamma_sum)
    constraints.append(P == Gamma_sum)
    
    prob = cp.Problem(cp.Maximize(cp.trace(P)), constraints)
    prob.solve(solver="CUOPT", verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("general_affine_inner_approx status:", prob.status)

    return P.value, pbar.value