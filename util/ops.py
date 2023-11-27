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


def get_param_types(param: dict):
    """
    A helper function to get types (integer or float) of param space. Returns a 
    dictionary with keys as param names, and values as param types ("int" or "float")
    `Param`: dict
      A dictionary with keys as parameter names, and values as parameter value range.
      If a parameter can only take on integer values, use only whole numbers (-3, 5, etc.)
      in range specification.
    """
    param_types = {}
    for name in param:
      if isinstance(param[name][0], int) and isinstance(param[name][1], int):
        param_types[name] = 'int'
      elif isinstance(param[name][0], float) and isinstance(param[name][1], float):
        param_types[name] = 'float'
      else:
        raise ValueError(f"Incompatible or inconsistent types detected for {name} parameter")
    return param_types

def get_scale_loc(unit_range, param_range):

    """
    A helper function to get scale and location shifter from unit range to param range,
    or vice versa. Both returned values are numpy arrays.
    """
    # supposed unit range is (a, b). param range is (c, d)
    # scale shifter = (d - c) / (b - a)
    # loc shifter = c - (1 + scale)*a

    scale_shift, loc_shift = [], []
    for i in range(len(param_range)):
      scale = (param_range[i][1] - param_range[i][0])/(unit_range[i][1] - unit_range[i][0])
      loc = param_range[i][0] - (1+scale)* unit_range[i][0]
      scale_shift.append(scale)
      loc_shift.append(loc)
    scale_shift = np.array(scale_shift)
    loc_shift = np.array(loc_shift)

    return scale_shift, loc_shift

def coor_change(coor, scale_shift, loc_shift, input_type, param_types):
    """
    A helper function to map the points between two coordinate system (spaces).
    Scale and loc shift are obtained through get_scale_loc() function.

    `coor`: point to be mapped
    `scale_shift`: scale shifters
    `loc_shift`: location shifters
    `input_type`: "unit" if coor is from unit range, "param" if coor is
    from param range
    `param_types`: types of parameter spaces. obtained through get_param_types()
    """
    # Map point (x) from unit range (a, b) to param range (c, d):
    # mapped_x = loc + scale * x
    # Map point (x) from param range (c, d) to unit range (a, b):
    # mapped_x = (x - loc) / scale
    if input_type == "unit":
      arr = loc_shift + scale_shift * coor 
    elif input_type == "param":
      arr = (coor - loc_shift) / scale_shift

    arr = arr.tolist()

    if input_type == 'unit':
      # this means we are mapping to param space, need t be cautious of param types (int or float)
      for i in range(len(param_types)):
        if list(param_types.values())[i] == "int":
          arr[i] = int(np.round(arr[i])) 
          # native int() method does not do proper rounding
          # np.round() does proper rounding, but will still return a float
          # chaining eliminates both issues

    return arr
