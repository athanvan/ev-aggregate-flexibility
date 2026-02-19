import numpy as np
import cvxpy as cp
from model_def_and_weights.model_weights import strip_model_data_new, create_A_matrix, construct_problem_data_cvxpy
from cvxpylayers.torch import CvxpyLayer

class CVXPY_Containment_Problem:
    """
    Implements Equation 10 as both a cvxpy problem and cvxpylayer 
    """

    def __init__(self, model, T, N, H_i, h_i, verbose = False, chosen_solver = "SCIPY"):
      self.N = N
      self.L = H_i[0].shape[0]
      self.T = T
      self.model = model
      self.verbose = verbose
      
      self.H_i = H_i
      self.h_i = h_i
      
      self.problem, self.parameter_list, self.variable_list = self.build_problem()
      self.layer = self.build_cvxpylayer_icnn()
      self.keywords = dict(verbose = True)
      
      # TODO: should I leave it like this or change it to 1e-6?
      self.cuopt_keywords = dict(
          absolute_primal_tolerance = 1e-4,
          relative_primal_tolerance = 1e-4,
          absolute_dual_tolerance = 1e-4, 
          relative_dual_tolerance = 1e-4
      )
      self.chosen_solver = chosen_solver
      
    def build_cvxpylayer_icnn(self):
      layer = CvxpyLayer(self.problem, parameters= self.parameter_list, 
                         variables= self.variable_list)
      return layer

    def build_problem(self):
      fc_weights, rl_weights, rl_biases, pmin, pmax = strip_model_data_new(self.model)
      fc_weight_params = []
      rl_weight_params = []
      rl_bias_params = []
      pmin_param = cp.Parameter(pmin.shape)
      pmax_param = cp.Parameter(pmax.shape)
      for i in range(len(fc_weights)):
          fc_weight_params.append(cp.Parameter(fc_weights[i].shape))
      for i in range(len(rl_weights)):
          rl_weight_params.append(cp.Parameter(rl_weights[i].shape))
          rl_bias_params.append(cp.Parameter(rl_biases[i].shape))

      C, d = construct_problem_data_cvxpy(self.model.input_dim, fc_weight_params, rl_weight_params, rl_bias_params, pmin_param, pmax_param)
      print(C.shape)
      print(d.shape)
      A = cp.Constant(create_A_matrix(self.T, C.shape[1]))
      print(A.shape)
      
      lambda_list = []
      beta_list = []
      gamma_list = []
      constraints = []
      sum1 = []
      sum2 = []
      scaling_factor = cp.Variable(bounds = [0.0, None], name = "scaling_factor")

      for idx in range(self.N):
        lam = cp.Variable((self.H_i[0].shape[0], C.shape[0]), pos = True, name = f"lam_{idx}")
        lambda_list.append(lam)
        beta = cp.Variable(self.T, name = f"beta_{idx}")
        beta_list.append(beta)
        gamma = cp.Variable((self.T, A.shape[1]), name = f"gamma_{idx}")
        gamma_list.append(gamma)

        constraints.append(lam @ C == self.H_i[idx] @ gamma)
        constraints.append(lam @ d <= self.h_i[idx] + self.H_i[idx] @ beta)
        sum1 += [-1 * beta]
        sum2 += [gamma]
        
      constraints.append(cp.sum(sum1) == cp.Constant(np.zeros(self.T)))
      constraints.append(cp.sum(sum2) == scaling_factor * A)

      obj = scaling_factor

      problem = cp.Problem(cp.Maximize(obj), constraints)
      assert problem.is_dpp()
      return problem, fc_weight_params + rl_weight_params + rl_bias_params + [pmin_param] + [pmax_param], [scaling_factor] + lambda_list + beta_list + gamma_list
      
    def solve_cvxpylayer_icnn(self):
      fc_weights, rl_weights, rl_biases, pmin, pmax = strip_model_data_new(self.model)
      try:
          sol = self.layer(
              *(fc_weights+rl_weights+rl_biases+[pmin]+[pmax]), solver_args = self.keywords)
      except Exception as e:
          raise e
      return sol
    
    def solve_icnn(self):
      fc_weights, rl_weights, rl_biases, pmin, pmax = strip_model_data_new(self.model)
      idx = 0
      for fc_w in fc_weights:
        self.parameter_list[idx].value = fc_w.cpu().detach().numpy()
        idx += 1
      for rl_w in rl_weights:
        self.parameter_list[idx].value = rl_w.cpu().detach().numpy()
        idx += 1
      for rl_b in rl_biases:
        self.parameter_list[idx].value = rl_b.cpu().detach().numpy()
        idx += 1
      self.parameter_list[idx].value = pmin.cpu().detach().numpy()
      idx += 1
      self.parameter_list[idx].value = pmax.cpu().detach().numpy()
      self.problem.solve(solver = self.chosen_solver, verbose = True, **self.cuopt_keywords)
      return self.problem.value, self.problem.status
