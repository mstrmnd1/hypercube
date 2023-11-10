from typing import Literal
import numpy as np
from copy import deepcopy
import itertools
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from hypercube.util.ops import check_param
from abc import ABC, abstractmethod


class base(ABC):

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
    self.param_type = check_param()
    self.random_state = random_state

  @abstractmethod
  def fit(self):
      pass


  # def _post_process(self) -> None:
  #   """
  #   `Overview`: 
  #   A private method for post-processing the experiment results. Logging
  #   key information as object attributes for easy access.
  #   """
  #   self.best_model = deepcopy(self.estimator).set_params(**self.best_param)
  #   self.best_model.fit(self.x, self.y)


  def predict(self, test_x):
    """
    `Overview`: 
    A public method for using the best model to predict y on new observation
    x. Recall that the best model is stored in self.best_model, and this estimator
    has already been trained in post-processing.
    """
    pred_y = self.best_model.predict(test_x)
    return pred_y



