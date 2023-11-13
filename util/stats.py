import numpy as np
from scipy import stats

def log_scaler(y):
    if np.min(y) <= 0:
        pass
    if np.max(y) <= 1:
        y = np.log(y)
    return y

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
    
def regression(x, y, intercept=True, demean=False):
    if demean:
      x = x - x.mean(axis=0)
    if intercept:
      x = np.hstack((np.ones((x.shape[0], 1)), x))
    beta = (np.linalg.inv(x.T @ x) @ x.T @ y)[:, np.newaxis]
    sse = np.linalg.norm((x @ beta).flatten() - y)**2
    mse = sse / (x.shape[0] - x.shape[1])
    var = mse * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(var))
    t_list = abs(beta.flatten()) / se
    df = x.shape[0] - x.shape[1]
    p_vals = [round((1 - stats.t.cdf(t_stat, df=df))*2, 4) for t_stat in t_list]
    return beta.flatten(), p_vals


def anova(design, y):

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
  for i in range(design.shape[1]):
    col = design[:, i]
    ss = 0
    uni_vals = np.unique(col)
    for v in uni_vals:
      idx = np.where(col == v)
      ss += (np.mean(y[idx]) - np.mean(y))**2 
    ss = n*ss*np.product(level_counts[np.arange(len(level_counts)) != i])
    MSS.append(ss)

  TSS = len(design)*np.var(y)
  RSS = TSS - sum(MSS)
  res_df = total_df - np.sum(model_df)
  mRSS = RSS / res_df
  f_stat = (np.array(MSS) / model_df) / mRSS
  p_val = [1 - stats.f.cdf(f, dfn=dfn, dfd=res_df) for f, dfn in zip(f_stat, model_df)]
  return {"model SS": np.round(MSS, 4), 
          "model df": model_df,
          "residual SS": np.round(RSS, 4),
          "residual df": res_df,
          "total SS": np.round(TSS, 4),
          "total df": total_df,
          "p values": np.round(p_val, 4)
          }
