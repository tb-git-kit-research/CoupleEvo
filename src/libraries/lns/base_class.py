import os
import importlib

from alns import ALNS
from alns import Result
import numpy.random as rnd

from types import FunctionType

import matplotlib.pyplot as plt

from solver.ceoh.utils.createFolders import *

import inspect

class LNS_Base:

    def __init__(self, problem: str):
        self.problem = problem

        # Empty classes for the module loading
        self._select = None
        self._accept = None
        self._stop = None
        self._initial_state = None
        self._load_data = None
        self._destroy = None
        self._repair = None
        self._info = None

        # Operators
        self._stop_operator = None
        self._select_operator = None
        self._accept_operator = None

        self._result = None
        self._init_sol = None

        self._data = None

        self._alns = ALNS(rnd.default_rng(seed=42))

        self.init()


    def init(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, "wrapper", f"{self.problem}.py")
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Cannot find module spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        save_lns_script(file_path)

        try:
            self._repair = getattr(module, "repair")
            self._destroy = getattr(module, "destroy")
            self._load_data = getattr(module, "load_data")
            self._initial_state = getattr(module, "initial_state")

            self._stop = getattr(module, "stop")
            self._accept = getattr(module, "accept")
            self._select = getattr(module, "select")
        except Exception as e:
            print(e)
            print("Please implement all neccassary functions: \n"
                  "repair / destroy \n"
                  "initial_state / load_data \n"
                  "stop / accept / select \n"
            )
            return

        if hasattr(module, "info"):
            self._info = getattr(module, "info")
        else:
            self._info = lambda _ : print("No Information")


    ####################################################################
    # LNS Core

    def init_lns(self,instance_seed = 0, paras = None, num_destroy = 1, num_repair = 1) -> None:

        if self._init_sol is None:
            data = self._load_data(instance_seed, paras)
            self._init_sol = self._initial_state(data, paras, instance_seed)
        self._alns.add_destroy_operator(self._destroy)
        self._alns.add_repair_operator(self._repair)

        self._select_operator = self._select(num_destroy, num_repair)
        self._accept_operator = self._accept(self._init_sol, paras.lns_paras["num_iters"])
        self._stop_operator = self._stop(paras.lns_paras["num_iters"], paras.lns_paras["runtime_limit"],paras.lns_paras["no_improvement_time"], paras.lns_paras["no_improvement_iters"])

    def run(self) -> None:
        self._result = self._alns.iterate(
                                self._init_sol,
                                self._select_operator,
                                self._accept_operator,
                                self._stop_operator)

    ####################################################################
    # Getter & Setter

    def set_destroy(self, destroy: FunctionType) -> None:
        self._destroy = destroy
        if self._alns:
            self._alns.add_destroy_operator(self._destroy)

    def set_repair(self, repair: FunctionType) -> None:
        self._repair = repair
        if self._alns:
            self._alns.add_repair_operator(self._repair)

    def set_data(self, data) -> None:
        if self._data is None:
            print("[LNS]: Loading data for the first time.")
        else:
            print("[LNS]: Overwriting data. Rerun the init!")

        self._data = data

    def get_result(self) -> Result:
        #print(f"Initial Objective: {self._init_sol.objective():<10}")
        #print(f"Initial Solution: {self._init_sol.solution}")
        #print(f"Final Objective: {self._result.best_state.objective():<10}")
        #print(f"Final Solution: {self._result.best_state.solution}")
        #print(self.get_statistics_summary())
        return self._result.best_state

    def get_statistics_summary(self) -> dict[str, float]:
        if self._result is None or self._result.statistics is None:
            return {}

        stats = self._result.statistics

        changes = []
        last_obj = None
        cum_runtime = 0.0
        n = len(stats.objectives)

        for i, (obj, rt) in enumerate(zip(stats.objectives, stats.runtimes)):
            cum_runtime += rt
            if i == 0 or obj != last_obj or i == n - 1:  # first, change, last
                changes.append({
                    "iteration": i,
                    "objective": float(obj),
                    "runtime_sec": float(cum_runtime),
                })
                last_obj = obj

        summary = {
            "total_runtime_sec": float(stats.total_runtime),
            "iterations": len(stats.objectives),
            "best_objective": float(min(stats.objectives)),
            "final_objective": float(stats.objectives[-1]),
            "initial_objective": float(self._init_sol.objective()) if self._init_sol else None,
            "initial_solution": self._init_sol.solution.tolist() if hasattr(self._init_sol.solution,
                                                                            "tolist") else self._init_sol.solution,
            "best_solution": self._result.best_state.solution,

            # Destroy operators
            "destroy_operator_names": list(stats.destroy_operator_counts.keys()),
            "destroy_operator_counts": list(stats.destroy_operator_counts.values()),
            "destroy_operator_weights": self._select_operator.destroy_weights.tolist(),

            # Repair operators
            "repair_operator_names": list(stats.repair_operator_counts.keys()),
            "repair_operator_counts": list(stats.repair_operator_counts.values()),
            "repair_operator_weights": self._select_operator.repair_weights.tolist(),

            # Settings
            "operator_coupling": self._select_operator.op_coupling.tolist(),
            "operator_weight_setting": self._select_operator.scores,
            "operator_decay_setting": self._select_operator.decay,
            "operator_num_repair_setting": self._select_operator.num_repair,
            "operator_num_destroy_setting": self._select_operator.num_destroy,

            # compressed instead of full arrays
            "improvement_log": changes,

        }

        return summary


    def get_objective_history(self) -> Result:
        return self._result.statistics

    ####################################################################
    # Visualization and Information

    def info(self) -> None:
        self._info(self._result )

    def plot_objective(self):
        _, ax = plt.subplots(figsize=(12, 6))

        if not self._result is None:
            self._result.plot_objectives(ax=ax)
            plt.show()

    def get_problem_class_string(self) -> str:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, "wrapper", f"{self.problem}.py")
        module = importlib.import_module(file_path)

        # Step 2: Get the class from the module
        class_name = 'ProblemClass'
        cls = getattr(module, class_name)

        # Step 3: Get the source code of the class as a string
        class_source = inspect.getsource(cls)

        print(class_source)