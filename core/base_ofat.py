import numpy as np
from typing import Literal
import itertools

from ..util.stats import t_test, regression, anova, z_scaler, log_scaler
from ..util.ops import run_cv, run_CVexp, get_design, get_combo, get_baseline_design
from .base_master import base

class OFAT(base):

  def __init__(self, estimator: object, param: dict, metric: str, 
               cv: int = 5, random_state: int = 0):
    """
    `Overview`:
    One-Factor-At-a-Time (OFAT) styled experiment object for hyperparamter 
    tuning. This is identical to the traditional grid search method. 

    OFAT only supports factors/hyperparamters with discrete levels/values. 

    `Args`:
    estimator: object
      A trained estimator (classification or regression) from sklearn. It is
      recommended to set random_state (if any) and any hyperparameters you'd
      like to constrain before loading.

      If evaluation metric involves predicting probabilities, then you need to
      set predict_proba=True.

    param: dict
      A dictionary of style {"parameter name": [possible values]}. In OFAT, each 
      parameter is constrained to discrete values only.

      To try several discrete values for a specific parameter, specify it as
      [value_1, value_2,..., value_k] using k-sized list. k >= 2.

    metric: str
      A string indicating the evaluation metric/objective to be optimized. If
      metric has inverse relationship with model performance (i.e. errors), use
      "neg_" prefix (e.g. neg_mean_squared_error). Call
      "sklearn.metrics.get_scorer_names()" to check all possible metric names.

    cv: int
      An integer representing the number of folds in K-Fold cross validation
      strategy.

    random_state: int or None
      A integer indicate the random state for sampling or K-Fold validation,
      if any. No random seed will be set if None was inputted.

    """
    super().__init__(estimator=estimator, param=param, metric=metric, 
                     cv=cv, random_state=random_state)
    if self.param_type != "discrete":
      raise ValueError("OFAT only supports discrete hyperparameter values")


  def fit(self, x: np.ndarray, y: np.ndarray, method: Literal['pair_t', 
                                                              'anova',
                                                              "lm_fit"],
          alpha: float = 0.05, scaling=None) -> None:
    """
    Main method for model fitting and tuning.

    `Args`:
    x: numpy array
      n*p matrix-like array with n observations and p features

    y: numpy array or list
      n-sized array with n target values

    method: literal['pair_t', 'anova'].
      'pair_t' is for paired t test approach.
      'anova' is for analysis of variance approach. 
    
    alpha: float
      A float value between (0, 1), indicating level of significance for t-tests or
      f-tests. Recommended values are betwwen (0.01, 0.1). 

    `Returns`:
    None
    """
    # assertions to check input types
    assert isinstance(x, np.ndarray) and (x.ndim == 2), \
      "x must be a two-dimensional numpy array"
    assert (isinstance(y, np.ndarray) and y.ndim == 1 and len(y) == len(x)) \
        or (isinstance(y, list) and len(y) == len(x)), \
      "y must be a numpy array or list of same length as x"
    assert method in ['pair_t', 'anova', "lm_fit"], \
      "method must be either 'pair_t' or 'anova'"
    assert isinstance(alpha, float) and (0 < alpha < 1), \
      "alpha value must take a float between 0 and 1"
    

    self.method = method
    self.init_score = run_cv(x, y, self.estimator, self.init_param, 
                             self.cv, self.scorer, self.random_state)
    self.best_score = self.init_score
    self.alpha = alpha

    if self.method == "pair_t":
      self._PairT(x, y, scaling)
    elif self.method == "anova":
      self._ANOVA(x, y)
    elif self.method == "lm_fit":
      self._LmFit(x, y)


  def _PairT(self, x, y, scaling) -> None:
    """
    `Overview`:
    A private method performing paired t-tests. 
    """
    self.design_mtx = get_design(self.param)
    self.combo = get_combo(self.param, self.design_mtx)
    exp_result = run_CVexp(x, y, estimator=self.estimator, param=self.combo,
                       cv=self.cv, scorer=self.scorer, 
                       random_state=self.random_state)

    for idx, new_score in enumerate(exp_result):
      if idx == 0:
        self.best_score = new_score
        self.best_param = self.combo[idx]
        continue

      if scaling == 'log':
        arr1 = log_scaler(self.best_score)
        arr2 = log_scaler(new_score)
      elif scaling == 'z':
        _ = z_scaler(np.concatenate((self.best_score, new_score)))
        midpoint = len(_) // 2
        arr1 = _[:midpoint]
        arr2 = _[midpoint:]
      else:
        arr1, arr2 = self.best_score, new_score

      if t_test(arr1=arr1, arr2=arr2, alternative="less", 
                alpha=self.alpha): 
        # True if new_score is better (greater) than best_score
        self.best_score = new_score
        self.best_param = self.combo[idx]

    print("Best hyperparameter combination:")
    print(self.best_param)

    
  def _ANOVA(self, x, y) -> None:
    self.design_mtx = get_design(self.param)
    self.combo = get_combo(self.param, self.design_mtx)
    exp_result = run_CVexp(x, y, estimator=self.estimator, param=self.combo,
                       cv=self.cv, scorer=self.scorer, 
                       random_state=self.random_state)
    loc = np.mean(exp_result, axis=1)
    disp = np.var(exp_result, axis=1)
    anova_loc = anova(self.design_mtx, z_scaler(loc))
    anova_disp = anova(self.design_mtx, log_scaler(disp))

    imp_idx = np.where(anova_loc['p values'] < self.alpha)[0].tolist()
    imp_factors = [list(self.param.keys())[i] for i in imp_idx]
    print(f"{imp_factors} are significant for location")
    
    imp_idx = np.where(anova_disp['p values'] < self.alpha)[0].tolist()
    imp_factors = [list(self.param.keys())[i] for i in imp_idx]
    print(f"{imp_factors} are significant for dispersion")

  def _LmFit(self, x, y) -> None:
    
    self.combo = get_combo(self.param, get_design(self.param))
    self.design_mtx, col_names = get_baseline_design(self.param)
    exp_result = run_CVexp(x, y, estimator=self.estimator, param=self.combo,
                       cv=self.cv, scorer=self.scorer, 
                       random_state=self.random_state)
    loc = np.mean(exp_result, axis=1)
    disp = np.var(exp_result, axis=1)
    beta, p_vals = regression(self.design_mtx, loc)
    print("For location:")
    print(f"Beta: {beta}, p values: {p_vals}")
    beta, p_vals = regression(self.design_mtx, log_scaler(disp))
    print("For dispersion:")
    print(f"Beta: {beta}, p values: {p_vals}")    

    
