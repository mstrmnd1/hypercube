import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
import itertools

def check_param(estimator, param):
    """
    `Overview`:
    A utility function checking parameter types.

    Recall that only lists of discrete values, tuples of continuous starting
    and ending points, and a mix of both are NOT supported. All other parameter
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
        raise ValueError("Mixed parameter types not supported")
    
  
def run_rep(train_x: np.ndarray, train_y: np.ndarray, 
            test_x: np.ndarray, test_y: np.ndarray, 
            estimator: object, scorer: object) -> list:
    """
    `Overview`:
      A function to run one single rep of training and testing (or
      validation). This function should be run with estimator already 
      loaded with experimenting hyperparameter

    `Args`:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    estimator: object
      A sklearn estimator (regressor or classifier)
    scorer: function
      A sklearn scorer 

    `Returns`:
    score: float
    """
    estimator.fit(train_x, train_y)
    return round(scorer(estimator, test_x, test_y), 4)
    

def run_cv(x: np.ndarray, y: np.ndarray, estimator: object, param_dict: dict,
           cv: int, scorer: object, random_state: int):
    """
    `Overview`:
      A function to run one set of cv, given a single set of hyperparameters.

    `Args`:
    x: np.ndarray
    y: np.ndarray
    estimator: object
      A sklearn estimator (regressor or classifier)
    param_dict: dict
      A combination of hyperparameter, with keys as parameter names and values
      as parameter values
    cv: int
    scorer: function
      A sklearn scorer
    random_state: int

    `Returns`:
    score: list
        k-sized list with validation scores across k folds. 
    """
    score, copy_estimator = [], deepcopy(estimator)
    copy_estimator.set_params(**param_dict)
    for train_idx, val_idx in KFold(n_splits=cv, shuffle=True,
                                    random_state=random_state).split(x, y):
        train_x, train_y = x[train_idx], y[train_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        score.append(run_rep(train_x, train_y, val_x, val_y, 
                             copy_estimator, scorer))
    return score


def run_CVexp(x: np.ndarray, y: np.ndarray, estimator: object, param: list,
              cv: int, scorer: object, random_state: int=1):
    """
    `Overview`:
      A function for running experiments. Under m combinations of hyperparameter, 
      this method runs k-fold cross-validation on x and y and returns a matrix 
      (m*k) of CV scores.

    `Args`:
    x: np.ndarray
    y: np.ndarray
    estimator: object
      sklearn estimator (regressor or classifier)
    param: list
      A list of dictionaries. Each dictionary is a combination of hyperparameter
      values. The keys are hyperparameter names and values are hyperparameter values.
    cv: int
    scorer: function
      sklearn scorer.
    random_state: int

    `Returns`:
    score: np.ndarray
      m*k sized matrix of m sets of hyperparameter combinations' performance
      across k folds. 
    """
    score = []
    for p in param:
      score.append(run_cv(x, y, estimator, p, cv, scorer, random_state))
    return np.array(score)


def get_design(param) -> np.ndarray:
    """
    Get regular design matrix of n*p size. 
    p = number of factors (type of hyperparameters)
    n = number of experiment runs. In full factorial setting, this is also 
    equivalent to the number of hyperparameter combinations. 
    """
    design_mtx = np.asarray(list(itertools.product(*param.values())), dtype=object)
    return design_mtx

def get_combo(param) -> list:
    """
    Get combinations of hyperparameters to be experimented.

    Returns:
    combo: list
      A list where each element in the list is a dictionary of one hyperparameter 
      combination. Such dictionary must be readable via sklearn's 
      estimator.set_params(**param). 
    """
    design_mtx = np.asarray(list(itertools.product(*param.values())), dtype=object)
    combo = [{k: v for k, v in zip(param.keys(), arr)} for arr in design_mtx]
    return combo


def get_baseline_design(param) -> (np.ndarray, np.ndarray):
    """
    Function to get qualitative/discrete variable encoding using baseline 
    constraints. Often used for regression analysis approach. 
    
    The encoding is similar to one-hot encoding or dummy variable encoding, 
    except that one level for each factor will be excluded. This is to ensure
    the design matrix is full-rank and thus invertible. 

    Returns:
    bsln_mtx: np.ndarray
      Baseline matrix
    col_names: np.ndarray
      Array of new column names
    """
    # get regular design matrix first (mtx)
    mtx = np.asarray(list(itertools.product(*param.values())), dtype=object)
    bsln_mtx = []
    col_names = []
    for i in range(mtx.shape[1]):
        # iterating through each factor
        param_name = list(param.keys())[i]
        col = mtx[:, i].astype(str) # change to string to avoid numpy errors
        uni_val = np.unique(col) # get distinct levels for the factor
        for j in range(1, len(uni_val)): # starting at idx 1 to exclude first level
            encoded = np.where(col == uni_val[j], 1, 0)
            bsln_mtx.append(encoded)
            new_name = param_name + "_" + uni_val[j] # create new column name
            col_names.append(new_name)
    bsln_mtx = np.array(bsln_mtx).T
    return bsln_mtx, np.array(col_names)