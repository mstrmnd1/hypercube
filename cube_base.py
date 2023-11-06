from typing import Literal
import numpy as np
from copy import deepcopy
import itertools
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class base:

  def __init__(self, estimator: object, param: dict, metric: str, 
               cv: int = 5, random_state: int = 0):
    """
    `Overview`:
    Base class for hypercube--a hyperparameter optimizer.

    Hypercube uses design of experiment (DOE) approaches to design and analyze
    hyperparameter experiments. Discrete, continuous or mixed hyperparameters
    are supported.

    `Args`:
    estimator: object
      A trained estimator (classification or regression) from sklearn. It is
      recommended to set random_state (if any) and any hyperparameters you'd
      like to constrain, before you load it to hypercube.

      If evaluation metric involves predicting probabilities, then you need to
      set predict_proba=True.

    param: dict
      A dictionary with names as keys and parameter ranges/options as values.
      Each parameter is constrained to either continuous or discrete values.

      To try a range of continuous values for a specific parameter, specify it
      as ({start_point}, {end_point}) using a two-sized tuple.

      To try several discrete values for a specific parameter, specify it as
      [{value1}, {value2}, {value3}] using k-sized list. k >= 2.

    metric: str
      A string indicating the evaluation metric/objective to be optimized. Call
      "sklearn.metrics.get_scorer_names()" to check all possible metric names.
      
      If metric has inverse relationship with model performance (i.e. errors), 
      use "neg_" prefix (e.g. neg_mean_squared_error). 

    cv: int
      An integer indicating the number of folds in K-Fold cross validation
      strategy.

    random_state: int or None
      A integer indicate the random state for sampling or K-Fold validation,
      if any. No random seed will be set if None was inputted.
    """
    assert isinstance(estimator, (BaseEstimator, ClassifierMixin, RegressorMixin)), \
      "estimator must be a scikit-learn classifier or regressor"
    assert isinstance(param, dict), \
      "param must be a dictionary"
    assert isinstance(metric, str), \
      "metric must be string"
    assert isinstance(cv, int), \
      "cv must be integer"
    assert isinstance(random_state, (int, type(None))), \
      "random_state must be an integer or None"

    self.estimator = estimator
    self.param = param
    if metric in metrics.get_scorer_names():
      self.scorer = metrics.get_scorer(metric)
    else:
      raise ValueError(f"Metric '{metric}' not found. Please use a string name from \
                       https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter")
    self.cv = cv
    self.init_param = estimator.get_params()
    self.param_type = self._check_param()
    self.random_state = random_state

  def _check_param(self):
    """
    `Overview`:
    A method checking parameter types.

    Recall that only lists of discrete values, tuples of continuous starting
    and ending points, and a mix of both are supported. All other parameter
    types will be rejected.

    `Return`:
    str: indicator of parameter space. "discrete", "continuous" or "mixed".
    """
    if all(name not in self.estimator.get_params() for name in self.param):
      # first check names of parameters
      raise ValueError("Misalignment on names of parameters")

    for name in self.param:
      # check requirements for lists and tuples
      if isinstance(self.param[name], list):
        continue
      elif isinstance(self.param[name], tuple):
        if len(self.param[name]) != 2:
          raise ValueError("Tuples must contain exactly 2 values, to indicate \
                           the starting and ending point of evaluation interval.")
        if all(isinstance(element, (int, float)) for element in self.param[name]) is False:
          raise ValueError("Tuples must contain numerical values only.")
      else:
          raise ValueError(f"Invalid data type for {name}. Use list for discrete values \
                           and tuple for continuous values.")

    # lastly check type of param space
    types = {type(self.param[name]) for name in self.param}
    if len(types) == 1:
        if list in types:
          return "discrete"
        elif tuple in types:
          return "continuous"
    elif len(types) == 2 and list in types and tuple in types:
        return "mixed"
    
  def _check_transform(self, y):
    if np.min(y) <= 0:
      pass
    if np.max(y) <= 1:
      y = np.log(y)
    return y
  
  def _run_rep(self, param_dict: dict):
    """
    `Overview`:
      A private method in the parent class for running one cross-validation 
      replication, under a fixed set of hyperparameter. 

    `Args`:
    param_dict: dict
      A dictionary of one set of hyperparamters for the estimator. 

    `Returns`:
    score: list
      k-sized list with validation scores across k folds. 
    """
    estimator = deepcopy(self.estimator)
    estimator.set_params(**param_dict)
    score = []
    for train_idx, val_idx in KFold(n_splits=self.cv, shuffle=True,
                                    random_state=self.random_state).split(self.x,
                                                                          self.y):
      train_x, train_y = self.x[train_idx], self.y[train_idx]
      val_x, val_y = self.x[val_idx], self.y[val_idx]
      estimator.fit(train_x, train_y)
      score.append(round(self.scorer(estimator, val_x, val_y), 3))
    return score


  def _run_experiment(self):
    """
    `Overview`:
      A private method in the parent class for running experiments. Under n
      combinations of hyperparameter, this method runs k-fold cross-validation
      on x and y and returns a matrix (n*k) of CV scores.

    `Args`:
    None. x and y values are obtained from obect attribute (self.x and self.y)

    `Returns`:
    score: numpy array
      n*k sized matrix of n sets of hyperparameter combinations' performance
      across k folds. 
    """
    score = []
    for p in self.combo:
      score.append(self._run_rep(param_dict=p))
    score = np.array(score)
    return score
  

  def _post_process(self) -> None:
    """
    `Overview`: 
    A private method for post-processing the experiment results. Logging
    key information as object attributes for easy access.
    """
    self.best_model = deepcopy(self.estimator).set_params(**self.best_param)
    self.best_model.fit(self.x, self.y)


  def predict(self, test_x):
    """
    `Overview`: 
    A public method for using the best model to predict y on new observation
    x. Recall that the best model is stored in self.best_model, and this estimator
    has already been trained in post-processing.
    """
    pred_y = self.best_model.predict(test_x)
    return pred_y



