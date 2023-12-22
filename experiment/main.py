import hyperopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..util.ops import get_combo, get_baseline_design
from ..util.stats import regression
from ..optimizer import problem, algo
import json


class trial:

    def __init__(self, prob, budget, dim) -> None:

        self.problem = prob
        self.func = problem.problem_all(prob)["func"]
        self.budget = budget
        if problem.problem_all(prob)["dim"] is None:
            self.dim = dim
        else:
            self.dim = problem.problem_all(prob)["dim"]
        self.bounds = problem.problem_all(prob)["bounds"]
        if len(self.bounds) == 1:
            self.bounds = self.bounds * self.dim

    def regret_plot(self):

        

        pass

    def solution_plot(self):

        X1, X2, Z = problem.truth(self.problem)
        plt.figure(figsize=(8, 6))
        cax = plt.pcolormesh(X1, X2, Z, shading='auto', cmap='viridis')
        plt.colorbar(cax)

        # for i in self.cube1_export:
        #     exp_dict = {"vanilla": {"x": self.cube1_export[i]["all_x"], "color": "red"},
        #             "moving": {"x": self.cube2_export[i]["all_x"], "color": "white"},
        #             "adj": {"x": self.cube3_export[i]["all_x"], "color": "orange"},
        #             "moving+adj": {"x": self.cube4_export[i]["all_x"], "color": "black"}}
        #     for exp in exp_dict:
        #         sol1 = exp_dict[exp]["x"][:, 0]
        #         sol2 = exp_dict[exp]["x"][:, 1]
        #         plt.scatter(sol1, sol2, c=exp_dict[exp]["color"], label=exp)
        #     plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.05))
        #     plt.show()

    def compar_plot(self):

        exp_attr = [attr for attr in vars(self) if 'export' in attr]
        res = [[l["best_res"] for l in list(getattr(self, name).values())] for name in exp_attr]
        plt.boxplot(res)
        plt.show()

    def analysis(self):

        x, col_names = get_baseline_design(self.dict1)
        col_names = np.insert(col_names, 0, "intercept")
        exp_attr = [attr for attr in vars(self) if 'export' in attr]
        arr = [[l["best_res"] for l in list(getattr(self, name).values())] for name in exp_attr]
        y = np.array(arr).mean(axis=1)
        summary = regression(x, y)
        summary.set_index(pd.Index(col_names), inplace=True)
        print(summary)
        

    def run_all(self, iter):
        
        self.iter = iter
        self._cube_temp("export1", adj_a=True, samp="ts", eff=3)
        self._tpe_run()


    def _rand_run(self):
        self.rand_export = {}
        for i in range(self.iter):    
            export = algo.RandSearch(self.func, obj="min", bounds=self.bounds, budget=self.budget)
            self.rand_export[i] = export

    def _lhs_run(self):
        self.lhs_export = {}
        for i in range(self.iter):    
            export = algo.Latin(self.func, obj="min", bounds=self.bounds, budget=self.budget)
            self.lhs_export[i] = export
    
    def _cube_temp(self, export_attr, adj_a, samp, eff):
        setattr(self, export_attr, {})
        for i in range(self.iter):
            export = algo.CuBE(self.func, obj="min", bounds=self.bounds, budget=self.budget,  
                               adj_a=adj_a, samp=samp, eff=eff)
            getattr(self, export_attr)[i] = export
        with open('data_cube.json', 'w') as file:
            json.dump(getattr(self, export_attr), file)
        delattr(self, export_attr)
    

    def algo_run(self, iter):

        dict1 = {"adj_a": [True, False],
                 "samp": ["ts", "opm", "ttts"],
                 "eff": [1, 3, 5]}
        self.dict1 = dict1
        self.iter = iter
        combo = get_combo(dict1)
        self.combo = combo
        for c in combo:
            val_list = [str(item) for item in list(c.values())]
            val_list.append("export")
            c["export_attr"] = '_'.join(val_list)
            self._cube_temp(**c)


    def _atpe_run(self):
        self._hyp_temp(hyperopt.atpe)

    def _tpe_run(self):
        self._hyp_temp(hyperopt.tpe)

    def _hyp_temp(self, method):

        def objective(x):
            input_x = list(x.values())
            return self.func(input_x)

        if len(self.bounds) == 1:
            space = {f"x{_}": hyperopt.hp.uniform(f"x{_}", self.bounds[0][0], 
                                                  self.bounds[0][1]) for _ in range(self.dim)}
        else:
            space = {}
            for i in range(len(self.bounds)):
                space[f"x{i}"] = hyperopt.hp.uniform(f"x{i}", self.bounds[i][0], 
                                                     self.bounds[i][1])

        name = method.__name__.split(".")[1] + "_export"
        setattr(self, name, {})
        for i in range(self.iter):
            trials = hyperopt.Trials()
            best = hyperopt.fmin(objective, space=space, algo=method.suggest,
                                 max_evals=self.budget, trials=trials)
            
            dict_exp = getattr(self, name, {})
            dict_exp[i] = {"best_x": list(best.values()), "best_res": np.min(trials.losses()),
                           "all_x": np.array(list(trials.vals.values())).T.tolist(), 
                           "all_res": list(trials.losses())}
            setattr(self, name, dict_exp)
        with open('data_hyp.json', 'w') as file:
            json.dump(getattr(self, name), file)
        delattr(self, name)

    
    def del_export(self):
        export_attrs = [attr for attr in self.__dict__ if 'export' in attr]
        for attr in export_attrs:
            delattr(self, attr)



class simulate:

    """
    studying the effect of budget, through large scale simulations
    """

    def __init__(self, problem):
        self.prob = problem

    def run(self, iter):
        return
