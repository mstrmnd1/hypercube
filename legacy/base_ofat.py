import numpy as np
import pandas as pd
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


  def fit(self, x: np.ndarray, y: np.ndarray, method: Literal['pair_t', 
                                                              'anova',
                                                              "lm_fit"],
          alpha: float = None) -> None:
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
      A float value between (0, 1), indicating level of significance in "pair_t" 
      approach Ignored if "anova" or "lm_fit" is selected. 
      Recommended values are betwwen (0.01, 0.1). 

    `Returns`:
    None
    """
    # assertions to check input types
    assert isinstance(x, np.ndarray) and (x.ndim == 2), \
      "x must be a two-dimensional numpy array"
    assert (isinstance(y, np.ndarray) and y.ndim == 1 and len(y) == len(x)), \
      "y must be a numpy array of same length as x"
    assert method in ['pair_t', 'anova', "lm_fit"], \
      "method must be either 'pair_t' or 'anova'"
    if method == "pair_t":
      assert isinstance(alpha, float), "alpha should be a float"
      assert 0 <= alpha <= 1, "alpha should be between 0 and 1"
      self.alpha = alpha
    
    self.method = method
    method_map = {
      "pair_t": self._PairT,
      "anova": self._ANOVA,
      "lm_fit": self._LmFit
    }
    if hasattr(self, "exp_result") is False:
      self.run(x, y)
    else:
      pass
    method_map[self.method]()


  def run(self, x, y) -> None:

    self.combo = get_combo(self.param)
    self.design_mtx = get_design(self.param)
    self.exp_result = run_CVexp(x, y, estimator=self.estimator, param=self.combo,
                       cv=self.cv, scorer=self.scorer, 
                       random_state=self.random_state)


  def _PairT(self) -> None:
    """
    `Overview`:
    A private method for t-test approach. 
    """
    # getting experiment results from design matrix

    for idx, new_score in enumerate(self.exp_result):
      if idx == 0:
        self.best_score = new_score
        self.best_param = self.combo[idx]
        continue

      if t_test(arr1=self.best_score, arr2=new_score, 
                alternative="less", alpha=self.alpha): 
        # True if new_score is better (greater) than best_score
        self.best_score = new_score
        self.best_param = self.combo[idx]


    
  def _ANOVA(self) -> None:
    """
    `Overview`:
    A private method for anova approach. 
    """    
    # getting experiment results from design matrix

    loc, disp = np.mean(self.exp_result, axis=1), np.var(self.exp_result, axis=1)
    indices = list(self.param.keys()) + ["Residual", "Total"]

    self._summary_ = {}
    # calling "anova" function to return anova table (pd dataframe)
    self._summary_["Location"] = anova(self.design_mtx, loc)
    self._summary_["Dispersion"] = anova(self.design_mtx, 
                                         log_scaler(disp)) # dispersion is log-scaled
    self._summary_["Location"].set_index(pd.Index(indices), inplace=True)
    self._summary_["Dispersion"].set_index(pd.Index(indices), inplace=True)

    self.best_score = self.exp_result[np.argmax(loc)]
    self.best_param = self.combo[np.argmax(loc)]
    

  def _LmFit(self) -> None:

    self.bsln_mtx, col_names = get_baseline_design(self.param)
    loc, disp = np.mean(self.exp_result, axis=1), np.var(self.exp_result, axis=1)

    self._summary_ = {}
    col_names = np.insert(col_names, 0, "intercept")
    # calling "regression" function to return regression table (pd dataframe)
    self._summary_["Location"] = regression(self.bsln_mtx, loc)
    self._summary_["Dispersion"] = regression(self.bsln_mtx, 
                                              log_scaler(disp)) # dispersion is log-scaled
    self._summary_["Location"].set_index(pd.Index(col_names), inplace=True)
    self._summary_["Dispersion"].set_index(pd.Index(col_names), inplace=True)

    self.best_score = self.exp_result[np.argmax(loc)]
    self.best_param = self.combo[np.argmax(loc)]
