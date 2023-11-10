from base.base_master import base

class Surf(base):

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
    