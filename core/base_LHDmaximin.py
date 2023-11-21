import numpy as np
import pandas as pd
from typing import Literal
from pyDOE import lhs
from scipy.spatial.distance import pdist, squareform

from ..util.stats import t_test, regression, anova, z_scaler, log_scaler
from ..util.ops import run_cv, run_CVexp, get_baseline_design
from .base_master import base

class LHSTuner(base):

    def __init__(self, estimator: object, param: dict, metric: str, 
                 cv: int = 5, random_state: int = 0, n_samples: int = 10):
        """
        Initialize the Latin Hypercube Sampling Tuner with maximin criterion.

        Args:
            estimator: The machine learning model or estimator.
            param: A dictionary where keys are parameter names and values are the ranges (tuples) for continuous parameters.
            metric: The performance metric to optimize.
            cv: The number of cross-validation folds.
            random_state: Random state for reproducibility.
            n_samples: The number of samples to generate in the Latin Hypercube.
        """
        super().__init__(estimator=estimator, param=param, metric=metric, cv=cv, random_state=random_state)
        if self.param_type != "continuous":
            raise ValueError("LHSTuner only supports continuous hyperparameter values")
        self.n_samples = n_samples

    def _generate_lhs_samples(self):
        """
        Generate Latin Hypercube Samples with maximin criterion.
        """
        lhs_samples = lhs(len(self.param), samples=self.n_samples, criterion="maximin")
        scaled_samples = np.array([np.linspace(start, end, self.n_samples) for (start, end) in self.param.values()])
        return lhs_samples * scaled_samples.T

    def fit(self, x: np.ndarray, y: np.ndarray, method: Literal['pair_t', 'anova', "lm_fit"], alpha: float = None) -> None:
        """
        Main method for model fitting and tuning, adjusted for LHS samples.

        Args:
            x: Feature matrix (numpy array).
            y: Target values (numpy array or list).
            method: Statistical method for analysis ('pair_t', 'anova', or 'lm_fit').
            alpha: Significance level for 'pair_t' method.
        """
        # Assertions to check input types
        assert isinstance(x, np.ndarray) and (x.ndim == 2), "x must be a two-dimensional numpy array"
        assert isinstance(y, np.ndarray) and y.ndim == 1 and len(y) == len(x), "y must be a numpy array of the same length as x"
        assert method in ['pair_t', 'anova', "lm_fit"], "method must be either 'pair_t', 'anova', or 'lm_fit'"
        if method == "pair_t":
            assert isinstance(alpha, float), "alpha should be a float"
            assert 0 <= alpha <= 1, "alpha should be between 0 and 1"
            self.alpha = alpha

        self.method = method
        method_map = {
            "pair_t": self._PairT,
            "anova": self._ANOVA,
            "lm_fit": self._LmFit
        }

    # Generate LHS samples
        lhs_samples = self._generate_lhs_samples()


    def _PairT(self) -> None:
        """
        `Overview`:
        A private method for t-test approach. 
        """
        # getting experiment results from design matrix

        for idx, new_score in enumerate(self.exp_result):
          if idx == 0:
            self.best_score = new_score
            self.best_param = self.combo[idx]
            continue

          if t_test(arr1=self.best_score, arr2=new_score, 
                alternative="less", alpha=self.alpha): 
            # True if new_score is better (greater) than best_score
            self.best_score = new_score
            self.best_param = self.combo[idx]


        
    def _ANOVA(self) -> None:
        """
        `Overview`:
        A private method for anova approach. 
        """    
        # getting experiment results from design matrix

        loc, disp = np.mean(self.exp_result, axis=1), np.var(self.exp_result, axis=1)
        indices = list(self.param.keys()) + ["Residual", "Total"]

        self._summary_ = {}
        # calling "anova" function to return anova table (pd dataframe)
        self._summary_["Location"] = anova(self.design_mtx, loc)
        self._summary_["Dispersion"] = anova(self.design_mtx, 
                                            log_scaler(disp)) # dispersion is log-scaled
        self._summary_["Location"].set_index(pd.Index(indices), inplace=True)
        self._summary_["Dispersion"].set_index(pd.Index(indices), inplace=True)

        self.best_score = self.exp_result[np.argmax(loc)]
        self.best_param = self.combo[np.argmax(loc)]
        

    def _LmFit(self) -> None:

        self.bsln_mtx, col_names = get_baseline_design(self.param)
        loc, disp = np.mean(self.exp_result, axis=1), np.var(self.exp_result, axis=1)

        self._summary_ = {}
        col_names = np.insert(col_names, 0, "intercept")
        # calling "regression" function to return regression table (pd dataframe)
        self._summary_["Location"] = regression(self.bsln_mtx, loc)
        self._summary_["Dispersion"] = regression(self.bsln_mtx, 
                                                log_scaler(disp)) # dispersion is log-scaled
        self._summary_["Location"].set_index(pd.Index(col_names), inplace=True)
        self._summary_["Dispersion"].set_index(pd.Index(col_names), inplace=True)

        self.best_score = self.exp_result[np.argmax(loc)]
        self.best_param = self.combo[np.argmax(loc)]
