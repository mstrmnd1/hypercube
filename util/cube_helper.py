import numpy as np
from math import floor

def update_grid(unit_x, success, alpha, beta, eff):
    coor = unit_to_idx(unit_x, alpha.shape[0])
    if success:
        alpha[coor] += eff
    else:
        beta[coor] += eff
    return alpha, beta

def update(obj, res, cutoff, unit_x, alpha, beta, eff):

    if obj == "max":
        if res >= cutoff:
            alpha, beta = update_grid(unit_x, True, alpha, beta, eff)
        elif res < cutoff:
            alpha, beta = update_grid(unit_x, False, alpha, beta, eff)
    elif obj == "min":
        if res >= cutoff:
            alpha, beta = update_grid(unit_x, False, alpha, beta, eff)                
        elif res < cutoff:
            alpha, beta = update_grid(unit_x, True, alpha, beta, eff)
    return alpha, beta


def OPM_TS(alpha, beta):

    theta_sample = np.random.beta(alpha, beta)
    mean_theta = alpha / (alpha + beta)
    max_mean = np.max(mean_theta)
    max_samp = np.max(theta_sample)
    if max_mean >= max_samp:
        idx = np.argmax(mean_theta)
    else:
        idx = np.argmax(theta_sample)
    return idx


def TTTS(alpha, beta, p=0.5, max_iter=50):
    
    theta_sample = np.random.beta(alpha, beta)
    max_idx = np.argmax(theta_sample)

    if np.random.binomial(1, p) == 1:
        return max_idx
    else:
        i = 0
        while i < max_iter:
            theta_sample = np.random.beta(alpha, beta)
            idx = np.argmax(theta_sample)
            i += 1
            if idx != max_idx:
                return idx
        return max_idx


def TS(alpha, beta):

    theta_sample = np.random.beta(alpha, beta)
    return np.argmax(theta_sample)

def sampling(samp, alpha, beta):

    if samp == "ts":
        idx = TS(alpha=alpha, beta=beta)
    elif samp == "opm":
        idx = OPM_TS(alpha=alpha, beta=beta)
    elif samp == "ttts":
        idx = TTTS(alpha=alpha, beta=beta)
    return idx

def unit_to_idx(unit, n):
    idx = []
    for p in unit:
        i = floor(p * n)
        if p == 1:
            i = n - 1
        idx.append(i)
    return np.array(idx)

def rem(bounds):

    p = len(bounds)
    low = np.array([tup[0] for tup in bounds])
    high = np.array([tup[1] for tup in bounds])
    A = np.random.random((3, p))
    new_bounds = [(l, h) for l, h in zip(A @ low, A @ high)]
    return A, new_bounds
