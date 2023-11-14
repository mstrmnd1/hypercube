import numpy as np
from typing import Literal

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
          alpha: float = 0.05) -> None:
    """
    Main method for model fitting and tuning.

    `Args`:
    x: numpy array
      n*p matrix-like array with n observations and p features

    y: numpy array or list
      n-sized array with n target values

    method: literal['pair_t', 'anova', 'lm_fit'].
      'pair_t' is for paired t test approach. This selects the best combination of
      hyperparameters, without any inference on individual/interaction effects.
      'anova' is for analysis of variance approach. Note that this can only provide
      inference on which hyperparameter is significant, not the direction of effect.
      'lm_fit' is the linear regression approach. This provides both inference on
      significant hyperparameters and the direction of effect. 
    
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
      self._PairT(x, y)
    elif self.method == "anova":
      self._ANOVA(x, y)
    elif self.method == "lm_fit":
      self._LmFit(x, y)


  def _PairT(self, x, y) -> None:
    """
    `Overview`:
    A private method for t-test approach. 
    """
    # getting experiment results from design matrix
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

      if t_test(arr1=self.best_score, arr2=new_score, 
                alternative="less", alpha=self.alpha): 
        # True if new_score is better (greater) than best_score
        self.best_score = new_score
        self.best_param = self.combo[idx]


    
  def _ANOVA(self, x, y) -> None:
    """
    `Overview`:
    A private method for anova approach. 
    """    
    # getting experiment results from design matrix
    self.design_mtx = get_design(self.param)
    self.combo = get_combo(self.param, self.design_mtx)
    exp_result = run_CVexp(x, y, estimator=self.estimator, param=self.combo,
                       cv=self.cv, scorer=self.scorer, 
                       random_state=self.random_state)
    
    loc = np.mean(exp_result, axis=1)
    disp = np.var(exp_result, axis=1)
    anova_loc = anova(self.design_mtx, loc)
    # dispersion will be log-scaled
    anova_disp = anova(self.design_mtx, log_scaler(disp))

    # finding important factors for location, based on specified alpha
    imp_idx = np.where(anova_loc['p values'] < self.alpha)[0].tolist()
    imp_factors = [list(self.param.keys())[i] for i in imp_idx]
    self.loc_factors = imp_factors
  
    # finding important factors for dispersion, based on specified alpha
    imp_idx = np.where(anova_disp['p values'] < self.alpha)[0].tolist()
    imp_factors = [list(self.param.keys())[i] for i in imp_idx]
    self.disp_factors = imp_factors
    self.anova_loc = anova_loc
    self.anova_disp = anova_disp


  def _LmFit(self, x, y) -> None:
    
    self.combo = get_combo(self.param, get_design(self.param))
    self.design_mtx, col_names = get_baseline_design(self.param)
    col_names = np.insert(np.array(col_names), 0, "intercept")

    exp_result = run_CVexp(x, y, estimator=self.estimator, param=self.combo,
                       cv=self.cv, scorer=self.scorer, 
                       random_state=self.random_state)
    self.all_result = exp_result
    self.loc = np.mean(exp_result, axis=1)
    self.disp = np.var(exp_result, axis=1)

    beta, p_vals, R_sqr = regression(self.design_mtx, self.loc)
    self.loc_fit = np.vstack([col_names, beta, p_vals]).T
    self.loc_R = R_sqr

    beta, p_vals, R_sqr = regression(self.design_mtx, log_scaler(self.disp))
    self.disp_fit = np.vstack([col_names, beta, p_vals]).T
    self.disp_R = R_sqr

    
