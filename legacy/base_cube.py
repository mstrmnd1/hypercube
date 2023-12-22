import numpy as np
import pandas as pd
from typing import Literal
from pyDOE import lhs
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from math import ceil, floor

from ..util.stats import t_test, regression, anova, z_scaler, log_scaler
from ..util.ops import run_cv, run_CVexp, get_param_types, coor_change, get_combo
from .base_master import base

class cube(base):

    def __init__(self, estimator: object, param: dict, metric: str, 
                 budget: int = 10, cv: int = 5, random_state: int = 0):
        """
        Cubic Bayesian Exploration

        Args:
            estimator: The machine learning model or estimator.
            param: A dictionary where keys are parameter names and values are the ranges (tuples) for continuous parameters.
            metric: The performance metric to optimize.
            cv: The number of cross-validation folds.
            random_state: Random state for reproducibility.
            n_samples: The number of samples to generate in the Latin Hypercube.
        """
        super().__init__(estimator=estimator, param=param, metric=metric, cv=cv, 
                         random_state=random_state)

        self.budget = budget
        self.loc_scale = list(self.param.values())
        
        samples = lhs(len(self.param), floor(self.budget/2), criterion="maximin")
        configs = []
        for row in samples:
            configs.append(self.coor_change(row, "unit"))
        self.combo = get_combo(self.param, design_mtx=configs)
        self.init_samples = samples
        self.res = []



    def coor_change(self, x, input_type):
        x = np.array(x)
        new = []
        n = len(x)
        loc_scale = list(self.param.values())
        if input_type == "unit":
            for i in range(n):
                new.append(loguniform.ppf(x[i], loc_scale[i][0], loc_scale[i][1]))
        elif input_type == "param":
            for i in range(n):
                new.append(loguniform.cdf(x[i],loc_scale[i][0], loc_scale[i][1]))
        return np.array(new)

        
    def update_grid(self, config, result):

        new = tuple()
        for coor in config:
            new += (ceil(coor * self.budget),)
        new = tuple(el - 1 for el in new)
        nbhd = tuple(slice(max(0, i-1), min(dim, i+2)) for i, dim in zip(new, self.alphas.shape))
        if isinstance(result, np.ndarray) or isinstance(result, list):
            for el in result:
                if el > self.cut_off:
                    self.alphas[nbhd] += 1
                    self.alphas[new] += self.cv - 1
                else:
                    self.betas[nbhd] += 1
                    self.betas[new] += self.cv - 1
        else:
            if result > self.cut_off:
                self.alphas[nbhd] += 1
                self.alphas[new] += self.cv - 1
            else:
                self.betas[nbhd] += 1
                self.betas[new] += self.cv - 1



    def thresholding(self, iter):

        # resource used = iter / budget (%)
        # set new cutoff = top % of existing data
        # retrive results from all visited grids
        # update alphas and betas for visited grids
        # for unvisited grids, perform normalization; total probability should be 
        return 



    def lhs_fit(self, x, y):

        samples = lhs(len(self.param), self.budget, criterion="maximin")
        configs = []
        for row in samples:
            configs.append(self.coor_change(row, "unit"))
        
        combo = get_combo(self.param, design_mtx=configs)
        results = run_CVexp(x, y, self.estimator, combo, cv=self.cv, 
                            scorer=self.scorer, random_state=self.random_state)
        results = np.mean(results, axis=1)
        best_idx = np.argmax(results)
        self.best_combo = combo[best_idx]
        self.best_score = results[best_idx]


    def fit(self, x, y):

        self.alphas = np.ones((self.budget, )*len(self.param))
        self.betas = np.ones((self.budget, )*len(self.param))
        results = run_CVexp(x, y, self.estimator, self.combo, cv=self.cv, 
                            scorer=self.scorer, random_state=self.random_state)
        results = np.mean(results, axis=1)
        self.res.extend(list(results))
        self.cut_off = np.median(results)
        for i, row in enumerate(results):
            self.update_grid(self.init_samples[i], row)
        
        for i in range(self.budget - len(results)):

            a_i = 1 - i / (self.budget - len(results))
            theta_sample = np.random.beta(self.alphas / a_i, self.betas / a_i)
            # regarding a, how about an adaptive approach? 
            # exploration favored when the algorithm has just begun
            # exploitation favored when nearing the end of iterations



            ### UPDATE: OPTIMISTIC SAMPLING
            mean_theta = self.alphas / (self.alphas + self.betas)
            max_mean = np.max(mean_theta)
            max_samp = np.max(theta_sample)
            if max_mean >= max_samp:
                idx = np.argmax(mean_theta)
            else:
                idx = np.argmax(theta_sample)
            ######

            next = np.unravel_index(idx, theta_sample.shape)
            unit_coor = (next + np.random.uniform(np.zeros(len(self.param)), np.ones(len(self.param)))) / self.budget
            p_coor = self.coor_change(unit_coor, "unit")
            new_combo = get_combo(self.param, design_mtx=[p_coor])[0]
            new_score = run_cv(x, y, self.estimator, new_combo, cv=3, scorer=self.scorer, 
                            random_state=self.random_state)
            self.update_grid(unit_coor, np.mean(new_score))
            self.res.append(np.mean(new_score))
            self.combo.append(new_combo)
        
        best_idx = np.argmax(self.res)
        self.best_combo = self.combo[best_idx]
        self.best_score = self.res[best_idx]


        
