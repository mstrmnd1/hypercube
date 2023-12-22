from sklearn.gaussian_process import GaussianProcessRegressor as gpr
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from skopt import optimizer

def gp_init(kernel, random_state):

    gp = gpr(kernel=kernel, random_state=random_state)
    return gp

def get_next_point(model, x, y, bounds):

    model.fit(x, y)
    x_try = x[np.random.choice(len(bounds)), :] + 0.1 * np.random.randn(x.shape[1])
    result = minimize(lambda xi: EI(xi, model), x0=x_try, bounds=bounds)
    next_point = result.x
    return next_point, model

def EI(x, model):

    mean, std = model.predict(x.reshape(1, -1), return_std=True)
    return -mean
    

