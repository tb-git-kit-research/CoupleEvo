import time
from copy import deepcopy
from dotenv import load_dotenv
from ortools.math_opt.python import solution

from libraries.lns.wrapper.lns_IRP_inventory import inventory_cost_from_shipments, routing_cost_from_routes
from libraries.lns.wrapper.lns_IRP_inventory import initial_state as inventory_initial_state
from libraries.lns.wrapper.lns_IRP_inventory import print_solution as inventory_print_solution
from libraries.lns.wrapper.lns_IRP_inventory import load_data as inventory_load_data

from solver.ceoh.utils.getParas import Paras

from libraries.lns.base_class import LNS_Base

import numpy.random as rnd

import os
import json

from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import NoImprovement

from libraries.lns.core.StopCriteria import MaxIterationsAndMaxRuntime, OrStop, NoImprovementTime

PROBLEM = "lns_IRP_vrp"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ProblemState:
    def __init__(self, solution, data, unassigned = []):
        self.solution = solution
        #"shipments": shipments,   # (T, V, N) NumPy array
        #"routes": routes          # dict[t][v] -> [(i,q), ...]
        self.data = data
        self.unassigned = unassigned
        self.parent_solution = copy.deepcopy(solution)

    def get_context(self):
        return

    def objective(self) -> float:
        total_costs = self.inventory_cost() + self.routing_cost()
        return total_costs

    def inventory_cost(self) -> float:
        return inventory_cost_from_shipments(self.solution["shipments"], self.data)

    def routing_cost(self):
        return routing_cost_from_routes(self.solution["routes"], self.data)

    def __str__(self):
        return f"Solution: {self.objective()} \n with routes: {self.solution}"



def initial_state(data, paras, instance_seed: int) -> ProblemState:
    return inventory_initial_state(data, paras, instance_seed)

####################################################################
# Repair (Random)

import copy
import numpy as np

def apply_scoring(state: ProblemState, scores, k: int) -> ProblemState:
    sol = state.solution
    shipments = sol["shipments"]
    routes = sol["routes"]

    # customers selected for removal
    to_remove = {i for i, _ in scores[:k]}

    for t in routes:
        for v in routes[t]:
            new_route = []
            for (i, q) in routes[t][v]:
                if i in to_remove:
                    # remove shipment
                    shipments[t, v, i] -= q
                    if shipments[t, v, i] < 1e-9:
                        shipments[t, v, i] = 0.0

                    # add to unassigned pool
                    state.unassigned.append((t, i, q))
                else:
                    new_route.append((i, q))

            routes[t][v] = new_route

    return state

def destroy(current: ProblemState, rng: rnd.Generator) -> ProblemState:
    destroyed = copy.deepcopy(current)
    scores,k = destroy_scoring(destroyed.solution, destroyed.data, rng)
    destroyed = apply_scoring(destroyed, scores, k)

    return destroyed

import numpy as np

def destroy_scoring(solution, data, rng):
    routes = solution["routes"]

    customers = set()
    for t in routes:
        for v in routes[t]:
            for i, _ in routes[t][v]:
                customers.add(i)

    if not customers:
        return [], 0

    customers = list(customers)
    rng.shuffle(customers)

    n = len(customers)
    low = max(1, int(0.05 * n))
    high = max(low, int(0.1 * n))
    k = int(rng.integers(low, high + 1))

    scores = [(i, rng.random()) for i in customers]
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores, k



####################################################################
# Repair
####################################################################
import time
def repair(destroyed: ProblemState, rng, max_time=10):
    start = time.perf_counter()


    def backtrack(state, remaining):
        if time.perf_counter() - start > max_time:
            timelimit_repaired = ProblemState(destroyed.parent_solution, destroyed.data)
            print("Return parent because of timeout. ")

            return timelimit_repaired

        if not remaining:
            return state

        item = remaining[0]
        rest = remaining[1:]

        options = get_insert_options(state, item)

        # small stochastic diversification
        #if len(options) > 1:
        #    head = options[:5]
        #    rng.shuffle(head)
        #    options = head + options[5:]

        for candidate in options:
            result = backtrack(candidate, rest)
            if result:
                return result

        return None

    repaired = backtrack(copy.deepcopy(destroyed), destroyed.unassigned)

    if repaired is None:
        raise RuntimeError("Repair failed: no feasible vrp reconstruction")
    repaired.unassigned = []
    return repaired

def get_insert_options(state, item):
    t, i, q = item
    options = []

    for v in state.solution["routes"][t]:
        route = state.solution["routes"][t][v]

        if not can_insert(q, route, state.data):
            continue

        for pos in range(len(route) + 1):
            cost = insert_cost(state, t, v, pos, i)

            new_state = copy.deepcopy(state)
            new_state.solution["routes"][t][v].insert(pos, (i, q))
            new_state.solution["shipments"][t, v, i] += q

            options.append((cost, new_state))

    options.sort(key=lambda x: x[0])
    return [s for _, s in options]



def insert_cost(state, t, v, pos, i):
    route = state.solution["routes"][t][v]
    C = state.data["cost_matrix"]

    prev_node = 0 if pos == 0 else route[pos - 1][0] + 1
    next_node = 0 if pos == len(route) else route[pos][0] + 1
    new_node  = i + 1

    return (
        C[prev_node][new_node]
        + C[new_node][next_node]
        - C[prev_node][next_node]
    )


def can_insert(q, route, data):
    cap = float(data["veh_cap"])
    used = sum(x for _, x in route)
    return used + q <= cap + 1e-9


####################################################################
# Utils
def load_data(instance_seed=0, paras=None):
    return inventory_load_data(instance_seed, paras)

def stop(num_iters : int, runtime_limit : int, no_improvement_time : int, no_improvement_iters : int):
    return OrStop(NoImprovement(no_improvement_iters), NoImprovementTime(no_improvement_time), MaxIterationsAndMaxRuntime(num_iters, runtime_limit))

def select(num_destroy: int, num_repair: int):
    return RouletteWheel([5, 2, 1, 0.5], 0.8, num_destroy, num_repair)

def accept(init: ProblemState, num_iters : int):
    return RecordToRecordTravel.autofit(
        init.objective(), 0.02, 0, num_iters
    )

def info(result):
    print("Info not implemented")

def print_solution(state: ProblemState):
    inventory_print_solution(state)


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    load_dotenv()
    #subprocess.run(["pip", "install", "."])

    os.environ["BASE_PATH"] = r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling"
    os.environ["INSTANCES_PATH"] = (
        r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling"
        r"\data\inventor_routing_instances\SKALNES\Instances"
    )

    start = time.time()

    print("Starting ALNS Route Optimization...")
    lns = LNS_Base(problem=PROBLEM)

    paras_instance = Paras()
    # add eoh_experiment_file to paras
    paras_instance.set_paras(
        eoh_experiment_file=f"U_C_N25",
        lns_paras = {
            "num_iters" : 100,
            "runtime_limit" : 10,
            "no_improvement_time" : 100,
            "no_improvement_iters" : 100,
        }
    )

    lns.init_lns(6, paras= paras_instance)

    lns.run()
    best_state = lns.get_result()
    statistics = lns.get_statistics_summary()
    print(best_state.objective())
    print("initial_objective ",statistics["initial_objective"])
    print("best_objective ",statistics["best_objective"])
    lns.plot_objective()
    result = lns.get_result()
    print(lns.get_statistics_summary())
    print("time passed", time.time() - start)



    best_state = lns.get_result()

    print("\n\n================ FINAL SOLUTION =================")
    print_solution(best_state)


    print("\n\n=================== OPTIMIZATION SUMMARY ===================")
    print("Initial Objective :", statistics["initial_objective"])
    print("Best Objective    :", statistics["best_objective"])
    print("Time Passed       :", time.time() - start)
