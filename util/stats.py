import numpy as np
from scipy import stats

def check_log(y):
    if np.min(y) <= 0:
        pass
    if np.max(y) <= 1:
        y = np.log(y)
    return y


def t_test(arr1, arr2, alternative, alpha):
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
    t_stat, p_val = stats.ttest_rel(arr1, arr2, alternative=alternative)
    if p_val <= alpha:
        return True
    else:
        return False
    
def lm_fit(x, y):

    x = np.hstack((np.ones((x.shape[0], 1)), x))
    beta = (np.linalg.inv(x.T @ x) @ x.T @ y)[:, np.newaxis]
    sse = np.linalg.norm((x @ beta).flatten() - y)**2
    mse = sse / (x.shape[0] - x.shape[1] - 1)
    var = mse * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(var))
    t_list = abs(beta.flatten()) / se
    df = x.shape[0] - x.shape[1]
    p_vals = [(1 - stats.t.cdf(t_stat, df=df))*2 for t_stat in t_list]