import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp 

def calculate_indiv_sets(a, d, N, T, seed, x_max_params = None, u_max_params = None, u_min_params = None):
    """
    Following flexibility set generation process as described in Appendix C
    """
    hi = np.zeros((4 * T, N), dtype=float)

    if x_max_params is None:
        x_max_params = [25, 50]
        u_max_params = [3, 10]
        u_min_params = [-10, -3]
        
    rng = np.random.default_rng(seed=seed)
    a = np.asarray(a).copy()
    d = np.asarray(d).copy()
    x_max = rng.uniform(x_max_params[0], x_max_params[1], size=N)
    u_max = rng.uniform(u_max_params[0], u_max_params[1], size=N)
    u_min = rng.uniform(u_min_params[0], u_min_params[1], size=N)
    x_init = rng.uniform(0, 0.4 * x_max)
    x_fin = rng.uniform(0.6 * x_max, x_max)

    for i in range(N):
        C_max = np.zeros(T)
        C_min = np.zeros(T)
        R_max = np.zeros(T)
        R_min = np.zeros(T)

        ai = int(np.floor(a[i]))
        di = int(np.ceil(d[i]))

        ai = max(1, min(T, ai))
        di = max(1, min(T, di))

        R_max[ai - 1:di] = u_max[i]
        R_min[ai - 1:di] = u_min[i]

        C_max[ai - 1:] = x_max[i] - x_init[i]

        if di - 1 >= ai:
            C_min[ai - 1:di - 1] = -x_init[i]
        C_min[di - 1:] = x_fin[i] - x_init[i]
        
        hi[:, i] = np.concatenate([C_max, -C_min, R_max, -R_min])
    return hi 

def find_chebyshev_center(A, b,verbose = False, chosen_solver = "SCS"):
  """"
  Finds the chebyshev center of some polytope Ax <= b
  """
  L = A.shape[0]
  T = A.shape[1]
  
  x = cp.Variable(T, name = "center")
  r = cp.Variable(nonneg = True, name = "r")
  norm = np.linalg.norm(A, axis = 1)
  constraints = [A @ x + r * norm <= b]
  #TODO: removed pmin and pmax constraints, is that fine?
  obj = cp.Maximize(r)
  problem = cp.Problem(obj, constraints)
  problem.solve(solver = chosen_solver, verbose = verbose)
  return problem