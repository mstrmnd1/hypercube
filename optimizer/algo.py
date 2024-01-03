import numpy as np
from math import ceil, floor
from time import time
from pyDOE import lhs
import matplotlib.pyplot as plt
from ..util.cube_helper import update, sampling, rem


def CuBE(func, obj, bounds, budget, adj_a=False, samp="ts", eff=1, n=10):
    
    p = len(bounds) # number of dimension
    alpha, beta = np.ones((n, )*p), np.ones((n, )*p)
    init_b = floor(budget/2)
    init_sample = lhs(n=p, samples=init_b, criterion="maximin")
    low = np.array([tup[0] for tup in bounds])
    high = np.array([tup[1] for tup in bounds])
    all_res, all_unit_x = [], []
    for unit_x in init_sample:
        x = low + (high - low) * unit_x
        all_res.append(func(x))
        all_unit_x.append(unit_x)

    cutoff = np.median(all_res)
    for res, unit_x in zip(all_res, all_unit_x):
        alpha, beta = update(obj, res, cutoff, unit_x, alpha, beta, eff=eff)

    for i in range(budget-init_b):
            
        if adj_a: a_i = 1 - i / (budget-init_b)
        else: a_i = 1
        idx = sampling(samp, alpha/a_i, beta/a_i)

        coor = np.unravel_index(idx, alpha.shape)
        unit_x = (coor + np.random.uniform(np.zeros(p), np.ones(p))) / n
        res = func(low + (high - low) * unit_x)
        all_res.append(res)
        all_unit_x.append(unit_x)

        perc = 100 * (1 - (i + init_b) / budget)
        cutoff = np.percentile(all_res, perc)
        for i, r in enumerate(all_res):
            alpha, beta = update(obj, r, cutoff, all_unit_x[i], 
                                     alpha, beta, eff=eff)

    all_x = low + (high - low) * np.array(all_unit_x)
    return end_algo(obj, all_res, all_x)



def RandSearch(func, obj, bounds, budget):

    p = len(bounds) # number of dimension
    low = np.array([tup[0] for tup in bounds])
    high = np.array([tup[1] for tup in bounds])
    all_res, all_x = [], []
    for _ in range(budget):
        x = np.random.uniform(np.zeros(p), np.ones(p))
        x = low + (high - low) * x
        all_res.append(func(x))
        all_x.append(x)
    return end_algo(obj, all_res, all_x)


def Latin(func, obj, bounds, budget):

    n = len(bounds)
    all_unit_x = lhs(n=n, samples=budget)
    low = np.array([tup[0] for tup in bounds])
    high = np.array([tup[1] for tup in bounds])
    all_res = []

    for unit_x in all_unit_x:
        x = low + (high - low) * unit_x
        all_res.append(func(x))

    all_x = low + (high - low) * np.array(all_unit_x)
    return end_algo(obj, all_res, all_x)


def end_algo(obj, all_res, all_x):

    if obj == "max":
        best_idx = np.argmax(all_res)
    elif obj == "min":
        best_idx = np.argmin(all_res)
    
    best_x = all_x[best_idx]
    best_result = all_res[best_idx]

    return {"best_x": list(best_x), "best_res": best_result
            # "all_x": all_x.tolist(), "all_res": all_res
            }




def CuBE2(func, obj, bounds, budget, adj_a=False, samp="ts", eff=1, n=10):
    
    A, bounds = rem(bounds)
    p = len(bounds) # number of dimension
    alpha, beta = np.ones((n, )*p), np.ones((n, )*p)
    init_b = floor(budget/2)
    init_sample = lhs(n=p, samples=init_b, criterion="maximin")
    low = np.array([tup[0] for tup in bounds])
    high = np.array([tup[1] for tup in bounds])
    all_res, all_unit_x = [], []
    for unit_x in init_sample:
        x = low + (high - low) * unit_x
        ori_x = (x.reshape(1, -1) @ A).flatten()
        all_res.append(func(ori_x))
        all_unit_x.append(unit_x)

    cutoff = np.median(all_res)
    for res, unit_x in zip(all_res, all_unit_x):
        alpha, beta = update(obj, res, cutoff, unit_x, alpha, beta, eff=eff)

    for i in range(budget-init_b):
            
        if adj_a: a_i = 1 - i / (budget-init_b)
        else: a_i = 1
        idx = sampling(samp, alpha/a_i, beta/a_i)

        coor = np.unravel_index(idx, alpha.shape)
        unit_x = (coor + np.random.uniform(np.zeros(p), np.ones(p))) / n
        x = (low + (high - low) * unit_x)
        ori_x = (x.reshape(1, -1) @ A).flatten()
        all_res.append(func(ori_x))
        all_unit_x.append(unit_x)

        perc = 100 * (1 - (i + init_b) / budget)
        cutoff = np.percentile(all_res, perc)
        for i, r in enumerate(all_res):
            alpha, beta = update(obj, r, cutoff, all_unit_x[i], 
                                     alpha, beta, eff=eff)

    all_x = low + (high - low) * np.array(all_unit_x)
    if obj == "max":
        best_idx = np.argmax(all_res)
    elif obj == "min":
        best_idx = np.argmin(all_res)
    
    best_x = all_x[best_idx]
    best_result = all_res[best_idx]

    return {"best_x": list(best_x), "best_res": best_result
            # "all_x": all_x.tolist(), "all_res": all_res
            }


