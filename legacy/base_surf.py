from .base_master import base
from ..util.ops import get_combo, run_CVexp, run_cv, get_2_interaction, get_full_design, get_pb12, get_param_types, coor_change
from ..util.stats import regression
import numpy as np
from scipy.stats import t
from scipy.optimize import minimize
from copy import deepcopy

class Surf(base):

  def __init__(self, estimator: object, param: dict, metric: str, 
               step_size: list, cv: int = 5, random_state: int = 0):
    """
    `Overview`:
    Response surface methods for hyperparameter tuning. 

    Surf only supports factors/parameters with continuous levels/values. 

    `Args`:
    estimator: object
      A trained estimator (classification or regression) from sklearn. It is
      recommended to set random_state (if any) and any hyperparameters you'd
      like to constrain before loading.

      If evaluation metric involves predicting probabilities, then you need to
      set predict_proba=True.

    param: dict
      A dictionary of style {"parameter name": [(loc, scale), dist]}. 
      In Surf, parameter type is constrained to continuous values only.

    metric: str
      A string indicating the evaluation metric/objective to be optimized. If
      metric has inverse relationship with model performance (i.e. errors), use
      "neg_" prefix (e.g. neg_mean_squared_error). Call
      "sklearn.metrics.get_scorer_names()" to check all possible metric names.

    step_size: list
      A list of n step sizes (learning rate) in n parameter spaces. If a step 
      size for a parameter is 0.05, that means the search grid will have 20 
      (1 / 0.05) points for that particular parameter. 

    cv: int
      An integer representing the number of folds in K-Fold cross validation
      strategy.

    random_state: int or None
      A integer indicate the random state for sampling or K-Fold validation,
      if any. No random seed will be set if None was inputted.

    """
    super().__init__(estimator=estimator, param=param, metric=metric, 
                     cv=cv, random_state=random_state)
    
    self.param_type = get_param_types(param)
    self.best_score = None
    self.score_log = []
    self.param_log = []
    self.method = "surf"
    self.unit_range = []
    for s in step_size:
      scale = 1 / s
      loc = - scale / 2
      self.unit_range.append((loc, scale))
    self.loc_scale = [t[0] for t in list(self.param.values())]
    self.dist_type = [t[1] for t in list(self.param.values())]
    self.step_size = np.array(step_size)


  def fit(self, x, y):

      centers = np.zeros(len(self.param))
      j = 0

      while j < 5: # five climbing opportunities at most
        j += 1
        prev_cen = deepcopy(centers)
        iter, centers = self._run_FO(x, y, centers)
        if centers is None: # this means bad fit from first order model
          self._run_SO(x, y, prev_cen)
          break

        if hasattr(self, "reach_bound"): # reached bound
          self._run_SO(x, y, centers)
          break

        if iter == 0: # climb not successful
          self._run_SO(x, y, centers)
          break

        if j == 5: # last chance to do second order
          self._run_SO(x, y, centers)
      

  def _run_FO(self, x, y, centers):
      CFOME = self._FO_design()
      nCFOME = CFOME + centers
      result = self._get_result(x, y, nCFOME)
      ascent = self._FO_analysis(CFOME, result)
      if ascent is None:
        return None, None
      iter, centers = self._climb(x, y, ascent=ascent, center=centers)
      return iter, centers

  def _run_SO(self, x, y, centers):
      CSOME = self._SO_design()
      nCSOME = CSOME + centers
      result = self._get_result(x, y, nCSOME)
      opt_p, _ = self._SO_analysis(CSOME, result)
      opt_p = opt_p + centers
      opt_p = coor_change(opt_p, param_type=self.param_type, loc_scale=self.loc_scale, 
                          dist_type=self.dist_type, input_type="unit", unit_range=self.unit_range)
      self.param_log.append(opt_p)
      self.best_param = {k: v for k, v in zip(self.param.keys(), opt_p)}
      best_score = run_cv(x, y, self.estimator, self.best_param, 
                          cv=self.cv, scorer=self.scorer, 
                          random_state=self.random_state)
      if self.cv != 1:
        self.best_score = np.mean(best_score)
      else:
        self.best_score = best_score
      self.score_log.append(self.best_score)

  def _FO_design(self, centers=None):

      design_mtx = get_full_design(len(self.param))
      if centers is None:
        centers = np.zeros(len(self.param))
        
      design_mtx = design_mtx + centers
      design_mtx = np.vstack((centers, design_mtx))
      return design_mtx
  
  def _FO_analysis(self, CFOME, result):
      """
      Get ascent direction for centered, first-order, main-effect design.
      """
      reg = regression(CFOME, result)
      coefs = np.array(reg["coef"])[1:]

      if (np.all(coefs == 0)) or (float(reg.at[0, 'R^2']) <= 0.4): # rare case
        return None
      
      ascent = coefs / np.max(np.abs(coefs))
      return ascent
  
  def _SO_design(self):
      
      """
      A centered design matrix, consisting of axial, corner and center points. 
      Technically, the terms are still first-order. To get quadratic terms, simply
      square the returned matrix.
      """
      p = len(self.param)
      axial = np.zeros((2*p, p))
      for i in range(p):
        axial[i, i] = -np.sqrt(p)  # Set 1 in the i-th position of the first p rows
        axial[i + p, i] = np.sqrt(p)  # Set -1 in the i-th position of the next p rows
      
      if (p == 2) or (p == 3) or (p == 1):
        corner = get_full_design(p)
      elif p == 4:
        pb12 = get_pb12()
        corner = np.delete(pb12, 2, axis=0)
        corner = pb12[:, :4]
      elif p == 5:
        pb12 = get_pb12()
        corner = np.delete(pb12, 2, axis=0)
        corner = pb12[:, [0, 1, 2, 3, 9]]

      SOME = np.vstack((axial, corner))
      SOME = np.vstack((np.zeros(p), SOME))
      return SOME


  def _SO_analysis(self, CSOME, result):
    """
    Get optimal (centered) unit coordinates for second-order experiment.

    CSOME: centered, second-order, main-effect design matrix. It should
    consist of axial, corner and center points.
    """
    p = len(self.param)
    design_mtx = get_2_interaction(CSOME)
    design_mtx = np.hstack((design_mtx, np.square(CSOME)))
    reg_result = np.array(regression(design_mtx, result)['coef'])
    coef = reg_result[1:]
    interc = reg_result[0]

    def objective_function(x):
      ints = []
      for i in range(len(x)):
        for j in range(i+1, len(x)):
          ints.append(x[i] * x[j])
      x = np.concatenate((x, np.array(ints), x))
      yhat = np.dot(x, coef) + interc
      return -yhat  # We use -y to convert the maximization problem to a minimization problem
  
    bounds = [(-1, 1)] * len(self.param) 
    init_guess = np.zeros(len(self.param))  
    result = minimize(objective_function, init_guess, bounds=bounds)
    opt_coor = result.x
    opt_result = -result.fun  # Convert back to maximize
    return opt_coor, opt_result
  

  def _climb(self, x, y, ascent, center):
      """
      Return optimal unit points from climbing FO ascent. 
      """
      i = 0
      while True:
        i += 1
        unit_points = center + i * ascent
        orig_points = coor_change(unit_points, param_type=self.param_type, loc_scale=self.loc_scale,
                                  dist_type=self.dist_type, input_type="unit", 
                                  unit_range=self.unit_range)
        self.param_log.append(orig_points)
        orig_points = {k: v for k, v in zip(self.param.keys(), orig_points)}  
        exp_result = run_cv(x, y, self.estimator, param_dict=orig_points, cv=self.cv, 
                            scorer=self.scorer, random_state=self.random_state)
        if self.cv != 1:
          exp_result = np.mean(exp_result)
        self.score_log.append(exp_result)

        if self.best_score is None:
          self.best_score = exp_result
        elif exp_result >= self.best_score:
          self.best_score = exp_result
        else: # time to change directions
          unit_points = unit_points - ascent # restore to optimum
          return i-1, unit_points
        
        ### sanity check: are the next set of points still feasible? ###
        next_unit_points = unit_points + ascent 
        end_point = 1 / self.step_size - 1 + np.sqrt(len(self.param)) # enough room for CCD
        if np.any((next_unit_points > end_point) | (next_unit_points < -end_point)):
          # boundary broken
          print("Local optimum reached, next point will be breaking boundary")
          self.reach_bound = True
          return i, unit_points
        ### end check ###
      
  def _get_result(self, x, y, nCME):
      """
      Get experiment result based on non-centered main-effect design matrix.
      """
      real_mtx = []
      for row in nCME:
        converted = coor_change(row, param_type=self.param_type, loc_scale=self.loc_scale,
                                dist_type=self.dist_type, input_type="unit",
                                unit_range=self.unit_range)
        real_mtx.append(converted)
      self.param_log.append(real_mtx)
      combo = get_combo(self.param, design_mtx=real_mtx)
      exp_result = run_CVexp(x, y, estimator=self.estimator, param=combo,
                             cv=self.cv, scorer=self.scorer, 
                             random_state=self.random_state) 
      if self.cv != 1:
        exp_result = np.mean(exp_result, axis=1)
      self.score_log.append(exp_result)
      return exp_result


      

    
