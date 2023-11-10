import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold

def check_param(estimator, param):
    """
    `Overview`:
    A utility function checking parameter types.

    Recall that only lists of discrete values, tuples of continuous starting
    and ending points, and a mix of both are supported. All other parameter
    types will be rejected.

    `Return`:
    str: indicator of parameter space. "discrete", "continuous" or "mixed".
    """
    if all(name not in estimator.get_params() for name in param):
      # first check names of parameters
      raise ValueError("Misalignment on names of parameters")

    for name in param:
      # check requirements for lists and tuples
      if isinstance(param[name], list):
        continue
      elif isinstance(param[name], tuple):
        if len(param[name]) != 2:
          raise ValueError("Tuples must contain exactly 2 values, to indicate \
                           the starting and ending point of evaluation interval.")
        if all(isinstance(element, (int, float)) for element in param[name]) is False:
          raise ValueError("Tuples must contain numerical values only.")
      else:
          raise ValueError(f"Invalid data type for {name}. Use list for discrete values \
                           and tuple for continuous values.")

    # lastly check type of param space
    types = {type(param[name]) for name in param}
    if len(types) == 1:
        if list in types:
          return "discrete"
        elif tuple in types:
          return "continuous"
    elif len(types) == 2 and list in types and tuple in types:
        return "mixed"
    

  
  
def run_rep(train_x, train_y, test_x, test_y, 
            estimator, param_dict: dict, scorer):
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
    estimator = deepcopy(estimator)
    estimator.set_params(**param_dict)
    estimator.fit(train_x, train_y)
    return round(scorer(estimator, test_x, test_y), 4)
    

def run_cv(x, y, estimator, param_dict, cv, scorer, random_state):

    score = []
    for train_idx, val_idx in KFold(n_splits=cv, shuffle=True,
                                    random_state=random_state).split(x, y):
        train_x, train_y = x[train_idx], y[train_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        score.append(run_rep(train_x, train_y, val_x, val_y, 
                             estimator, param_dict, scorer))
    return score

def run_CVexp(x, y, estimator, param, cv, scorer, random_state=1):
   
    """
    `Overview`:
      A private method in the parent class for running experiments. Under n
      combinations of hyperparameter, this method runs k-fold cross-validation
      on x and y and returns a matrix (n*k) of CV scores.

    `Args`:
    

    `Returns`:
    score: numpy array
      n*k sized matrix of n sets of hyperparameter combinations' performance
      across k folds. 
    """
    score = []
    for p in param:
      score.append(run_cv(x, y, estimator, p, cv, scorer, random_state))
    return np.array(score)


