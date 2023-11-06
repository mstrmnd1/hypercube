from base.cube_base import base
import numpy as np
from typing import Literal
import itertools
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

class _ofat_(base):

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

  def fit(self, x: np.ndarray, y: np.ndarray, method: Literal['pair_t', 'anova'],
          alpha: float = None) -> None:
    """
    Main method for model fitting and tuning.

    `Args`:
    x: numpy array
      n*p matrix-like array with n observations and p features

    y: numpy array or list
      n-sized array with n target values

    method: literal['pair_t', 'anova'].
      'pair_t' is for paired t test approach. Must specify alpha value if selected.
      'anova' is for analysis of variance approach. 
    
    alpha: float
      A float value between (0, 1), indicating level of significance for t tests. 
      Must specify alpha value, if 'pair_t' is selected as the method. Recommended
      values for alpha are betwwen (0.01, 0.1). 

    `Returns`:
    None
    """
    # assertions to check input types
    assert isinstance(x, np.ndarray) and (x.ndim == 2), \
      "x must be a two-dimensional numpy array"
    assert (isinstance(y, np.ndarray) and y.ndim == 1 and len(y) == len(x)) \
        or (isinstance(y, list) and len(y) == len(x)), \
      "y must be a numpy array or list of same length as x"
    assert method in ['pair_t', 'anova'], \
      "method must be either 'pair_t' or 'anova'"
    if method == "pair_t":
        assert isinstance(alpha, float) and (0 < alpha < 1), \
          "when method='pair_t', alpha value must take a float between 0 and 1"
    
    self.x = x
    self.y = y
    self.method = method
    self.init_score = self._run_rep(self.init_param)
    self.best_score = self.init_score

    if self.method == "pair_t":
      self.alpha = alpha
      self._pair_t()
    elif self.method == "anova":
      return self._anova()
    # self._post_process()


  def _pair_t(self) -> None:
    """
    `Overview`:
    A private method performing paired t-tests. 
    """
    design_mtx = np.array(list(itertools.product(*self.param.values())))
    combo = [{k: v for k, v in zip(self.param.keys(), arr)} for arr in design_mtx]
    self.design_mtx = design_mtx
    self.combo = combo
    result = self._run_experiment()

    for idx, new_score in enumerate(result):
      if self._t_test(self.best_score, new_score): 
          # True if score is better than best_score
          self.best_score = new_score
          self.best_param = combo[idx]

  def _t_test(self, arr1, arr2):
    """
    `Overview`:
      A private method used by _pair_t() function. It performs a paired t-test
      on two score arrays, under user-defined alpha.

    `Args`:
    arr1: numpy.ndarray
      a k-sized array of current best scores
    arr2: numpy.ndarray
      a k-sized array of scores to be compared

    `Returns`:
    bool: True if arr2 is better than arr1. False otherwise.
      Note that arr2 is better than arr1 when p value is less than alpha, under 
      the "less" alternative in scipy.stats.ttest_rel(). 
      This is because, in all sklearn scorers, greater is always better (including
      negative error matrics: neg_mean_squared_error). 
    """
    t_stat, p_val = stats.ttest_rel(arr1, arr2, alternative="less")
    if p_val <= self.alpha:
        return True
    else:
        return False
    
  def _anova(self) -> None:

    design_mtx = np.array(list(itertools.product(*self.param.values())))
    combo = [{k: v for k, v in zip(self.param.keys(), arr)} for arr in design_mtx]
    self.design_mtx, cols = self._one_hot(design_mtx.astype(str))
    self.combo = combo
    cols = [list(combo[0].keys())[index] + "_" + element for index, sublist in 
            enumerate(cols) for element in sublist]
    result = self._run_experiment()
    print(result)
    # one_hot = pd.DataFrame(self.design_mtx, columns=cols)
    # result = pd.concat([one_hot, result], axis=1)
    return self._linear_fit(design_mtx=self.design_mtx, result=result)

    # analyze result of experiment

  def _one_hot(self, design_mtx):
    one_hot = []
    cols = []
    for i in range(design_mtx.shape[1]):
        unique_values = np.unique(design_mtx[:, i])
        encoded_column = np.zeros((design_mtx.shape[0], len(unique_values)))
        for j, value in enumerate(design_mtx[:, i]):
            encoded_column[j, np.where(unique_values == value)[0][0]] = 1
        one_hot.append(encoded_column)
        cols.append(unique_values.tolist())
    one_hot = np.hstack(one_hot)
    return one_hot, cols
  
  
  def _linear_fit(self, x, y):
     
    y_bar = np.mean(y, axis=1)
    # y_bar = self._check_transform(y_bar)
    x = sm.add_constant(x)
    model = smf.ols("y ~ x", data={"y": y_bar, "x": x}).fit()
    # anova_table = sm.stats.anova_lm(model, typ=2)
    # print(anova_table)
    return model
  

     




class _surf_(base):

  def __init__(self, estimator: object, param: dict, metric: str, 
               cv: int = 5, random_state: int = 0):
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
      A dictionary of style {"parameter name": (start, end)}. In Surf, parameter
      type is constrained to continuous values only.

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
    if self.param_type != "continuous":
      raise ValueError("Surf only supports continuous hyperparameter values")

  def 
    
    