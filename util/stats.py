import numpy as np
import pandas as pd
from scipy import stats

def log_scaler(y):
    
    if np.min(y) <= 0:
       y[y <= 0] = np.min(y[y > 0])
  
    return np.log(y)

def z_scaler(data):
    
    if isinstance(np.std(data, axis=0), np.ndarray):
       if 0 in np.std(data, axis=0):
          pass
       else:
          data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    else: 
       if np.std(data, axis=0) == 0:
          pass
       else:
          data = (data - np.mean(data)) / np.std(data)
    return data


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
    if np.isnan(p_val):
       return False
  
    if p_val <= alpha :
        return True
    else:
        return False
    

def regression(x, y, intercept=True):

    """
    perform linear regression, return regression table
    """
    if intercept:
      x = np.hstack((np.ones((x.shape[0], 1)), x))
    beta = (np.linalg.inv(x.T @ x) @ x.T @ y)[:, np.newaxis]

    TSS = np.linalg.norm(y - np.mean(y)) ** 2
    RSS = np.linalg.norm((x @ beta).flatten() - y)**2
    mRSS = RSS / (x.shape[0] - x.shape[1])

    beta_se = np.sqrt(np.diag(mRSS * np.linalg.inv(x.T @ x)))
    t_list = abs(beta.flatten()) / beta_se

    df = x.shape[0] - x.shape[1]
    p_vals = np.array([round((1 - stats.t.cdf(t_stat, df=df))*2, 4) 
                      for t_stat in t_list])
    R_sqr = 1 - RSS/TSS

    summary = pd.DataFrame({"coef": beta.flatten(), "p_val": p_vals, 
                            "R^2": ""})
    summary.at[0, "R^2"] = R_sqr
    return summary


def anova(design, y):
    """
    perform anova for qualitative factors, return anova table
    """
    design = design.astype(str)
    if (np.ndim(y) == 1) or (1 in y.shape):
      n = 1
    else:
      assert y.shape[0] == design.shape[0]
      n = y.shape[1]
      y = np.hstack(y)
      design = np.repeat(design, n, axis=0)
    
    def count_unique(column):
      return len(np.unique(column))

    level_counts = np.apply_along_axis(count_unique, axis=0, arr=design)
    model_df = level_counts - 1
    total_df = len(design) - 1
    MSS = []
    # below is calculating the model sum of squares
    # this can probably to optimized :)
    for i in range(design.shape[1]):
      col = design[:, i]
      ss = 0
      uni_vals = np.unique(col)
      for v in uni_vals:
        idx = np.where(col == v)
        ss += (np.mean(y[idx]) - np.mean(y))**2 
      ss = n*ss*np.product(level_counts[np.arange(len(level_counts)) != i])
      MSS.append(ss)

    TSS = np.linalg.norm(y - np.mean(y))**2
    RSS = TSS - sum(MSS)
    res_df = total_df - np.sum(model_df)
    f_stat = (np.array(MSS) / model_df) / (RSS / res_df)
    p_val = [1 - stats.f.cdf(f, dfn=dfn, dfd=res_df) for f, dfn in zip(f_stat, model_df)]

    summary = pd.DataFrame(columns=["SS", "df", "p-val"], 
                      index=range(len(p_val)+2)) # len(p_val) is the number of factors
    summary["SS"][:len(p_val)] = np.round(MSS, 4)
    summary["df"][:len(p_val)] = model_df
    summary['p-val'][:len(p_val)] = np.round(p_val, 4)
    summary.at[len(p_val), "SS"] = np.round(RSS, 4)
    summary.at[len(p_val), "df"] = res_df
    summary.at[len(p_val)+1, "SS"] = np.round(TSS, 4)
    summary.at[len(p_val)+1, "df"] = total_df
    return summary

