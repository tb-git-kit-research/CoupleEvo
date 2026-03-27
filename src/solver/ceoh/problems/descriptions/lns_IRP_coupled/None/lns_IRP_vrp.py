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

def apply_scoring(day_routes, day_shipments, t, scores, k, state):
    """
    day_routes: dict[v] -> [(i, q), ...]
    day_shipments: shipments[t]
    """

    to_remove = {i for i, _ in scores[:k]}

    for v in day_routes:

        new_route = []

        for (i, q) in day_routes[v]:

            if i in to_remove:

                # update shipment tensor
                day_shipments[v, i] -= q
                if day_shipments[v, i] < 1e-9:
                    day_shipments[v, i] = 0.0

                # add to unassigned pool
                state.unassigned.append((t, i, q))

            else:
                new_route.append((i, q))

        day_routes[v] = new_route


def destroy(current: ProblemState, rng: rnd.Generator) -> ProblemState:

    destroyed = copy.deepcopy(current)

    routes = destroyed.solution["routes"]
    shipments = destroyed.solution["shipments"]

    for t in routes:

        day_routes = routes[t]
        day_shipments = shipments[t]

        scores, k = destroy_scoring(day_routes, destroyed.data, rng)

        if k > 0:
            apply_scoring(day_routes, day_shipments, t, scores, k, destroyed)
    return destroyed


def destroy_scoring(day_routes, data, rng):
    # Collect unique customers served today
    customers = {
        i
        for v in day_routes
        for (i, _) in day_routes[v]
    }

    if not customers:
        return [], 0

    customers = list(customers)
    n = len(customers)

    # Remove between 5% and 15% of active customers
    low = max(1, int(0.05 * n))
    high = max(low, int(0.15 * n))

    k = int(rng.integers(low, high + 1))

    # Direct random selection (no scoring needed)
    selected = rng.choice(customers, size=k, replace=False)

    # Apply_scoring expects (i, score) pairs
    return [(int(i), 1.0) for i in selected], k



####################################################################
# Repair
####################################################################
import time

def repair(destroyed: ProblemState, rng, max_time=1, beam_width=5):

    # Work on a copy
    state = copy.deepcopy(destroyed)

    # Nothing to repair
    if not state.unassigned:
        return state

    # -------------------------------------------------
    # Group unassigned by day
    # -------------------------------------------------
    unassigned_by_day = {}

    for (t, i, q) in state.unassigned:
        unassigned_by_day.setdefault(t, []).append((t, i, q))

    # Clear global unassigned list
    state.unassigned = []

    # -------------------------------------------------
    # Repair each day independently
    # -------------------------------------------------
    for t in sorted(unassigned_by_day.keys()):

        ok = repair_day(
            state=state,
            t=t,
            items=unassigned_by_day[t],
            rng=rng,
            max_time=max_time,
            beam_width=beam_width
        )

        if not ok:
            # fallback to parent solution
            return ProblemState(state.parent_solution, state.data)

    return state

def repair_day(state, t, items, rng, max_time, beam_width):

    start = time.perf_counter()
    deadline = start + float(max_time)

    routes = state.solution["routes"][t]
    shipments = state.solution["shipments"][t]

    V = int(state.data["V"])
    cap = float(state.data["veh_cap"])
    C = state.data["cost_matrix"]

    loads = np.zeros(V)
    for v in range(V):
        loads[v] = sum(q for _, q in routes[v])

    # --- delta cost ---
    def insert_delta_cost(v, pos, cust_i):
        route = routes[v]
        prev_node = 0 if pos == 0 else route[pos - 1][0] + 1
        next_node = 0 if pos == len(route) else route[pos][0] + 1
        new_node = cust_i + 1
        return (
            C[prev_node][new_node]
            + C[new_node][next_node]
            - C[prev_node][next_node]
        )

    def iter_best_moves(cust_i, qty_q):
        moves = []

        for v in range(V):
            if loads[v] + qty_q > cap + 1e-9:
                continue

            route = routes[v]
            for pos in range(len(route) + 1):
                dc = insert_delta_cost(v, pos, cust_i)
                moves.append((dc, v, pos))

        if not moves:
            return []

        moves.sort(key=lambda x: x[0])
        return moves[:beam_width]

    def apply_move(v, pos, cust_i, qty_q):
        routes[v].insert(pos, (cust_i, qty_q))
        shipments[v, cust_i] += qty_q
        loads[v] += qty_q

    def undo_move(v, pos, cust_i, qty_q):
        routes[v].pop(pos)
        shipments[v, cust_i] -= qty_q
        loads[v] -= qty_q

    items = sorted(items, key=lambda x: -x[2])  # large first

    def backtrack(idx):
        if time.perf_counter() > deadline:
            return False

        if idx >= len(items):
            return True

        _, cust_i, qty_q = items[idx]
        moves = iter_best_moves(cust_i, qty_q)

        if not moves:
            return False

        for _, v, pos in moves:
            apply_move(v, pos, cust_i, qty_q)
            if backtrack(idx + 1):
                return True
            undo_move(v, pos, cust_i, qty_q)

        return False

    return backtrack(0)

"""
import copy
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def repair(current: ProblemState, rng=None, max_time=100):
    print("\n================ ORTOOLS LOCKED ROUTE REPAIR START ================")

    repaired = copy.deepcopy(current)
    data = repaired.data

    N = int(data["N"])
    T = int(data["T"])
    V = int(data["V"])
    cap = float(data["veh_cap"])
    C = np.array(data["cost_matrix"])

    repaired.solution["routes"] = {t: {v: [] for v in range(V)} for t in range(T)}
    repaired.solution["shipments"] = np.zeros_like(current.solution["shipments"])

    for t in range(0, T):   # your test day
        print(f"\n--- Day {t} ---")

        # -------------------------------------------------
        # Build nodes EXACTLY from existing routes
        # -------------------------------------------------
        nodes = []
        initial_routes = [[] for _ in range(V)]

        for v in range(V):
            for (cust, q) in current.solution["routes"][t][v]:
                if q <= 1e-9:
                    continue
                nodes.append((cust, float(q)))
                node_id = len(nodes)  # 1-based (0 is depot)
                initial_routes[v].append(node_id)

        M = len(nodes)

        if M == 0:
            print(f"[DAY {t}] No active nodes → copying original")
            repaired.solution["routes"][t] = copy.deepcopy(current.solution["routes"][t])
            repaired.solution["shipments"][t] = current.solution["shipments"][t].copy()
            continue

        print(f"[DAY {t}] Total nodes: {M}")

        # -------------------------------------------------
        # Build Routing Model
        # -------------------------------------------------
        manager = pywrapcp.RoutingIndexManager(M + 1, V, 0)
        routing = pywrapcp.RoutingModel(manager)

        # ----- Cost callback -----
        def cost_cb(from_index, to_index):
            fn = manager.IndexToNode(from_index)
            tn = manager.IndexToNode(to_index)

            fr = 0 if fn == 0 else nodes[fn - 1][0] + 1
            to = 0 if tn == 0 else nodes[tn - 1][0] + 1

            return int(C[fr][to])

        transit = routing.RegisterTransitCallback(cost_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit)

        # ----- Capacity -----
        scale = 1
        int_cap = int(round(cap * scale))
        int_demands = [0] + [int(round(q * scale)) for (_, q) in nodes]

        def demand_cb(from_index):
            return int_demands[manager.IndexToNode(from_index)]

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx,
            0,
            [int_cap] * V,
            True,
            "Capacity"
        )

        # -------------------------------------------------
        # CLOSE MODEL BEFORE APPLYING LOCKS
        # -------------------------------------------------
        routing.CloseModel()

        # -------------------------------------------------
        # APPLY LOCKS (FULL FREEZE)
        # -------------------------------------------------
        #use_lock = False
        #if use_lock:
        #    ok = routing.ApplyLocksToAllVehicles(initial_routes, True)
        #    if not ok:
        #        print(f"[DAY {t}] ❌ Unable to apply locks → reverting")
        #        repaired.solution["routes"][t] = copy.deepcopy(current.solution["routes"][t])
        #        repaired.solution["shipments"][t] = current.solution["shipments"][t].copy()
        #        continue

        # Use PreAssignment (correct API for locks)
        #assignment = routing.PreAssignment()
        assignment = routing.ReadAssignmentFromRoutes(initial_routes, True)

        # -------------------------------------------------
        # Solve from locked assignment
        # -------------------------------------------------
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        params.time_limit.FromSeconds(max_time)
        params.log_search = True

        sol = routing.SolveFromAssignmentWithParameters(assignment, params)
        #sol = routing.SolveWithParameters(params)

        status = routing.status()

        if sol is None:
            print(f"[DAY {t}] ❌ Solver failed")
            print("Routing status code:", status)

            status_map = {
                0: "ROUTING_NOT_SOLVED",
                1: "ROUTING_SUCCESS",
                2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
                3: "ROUTING_FAIL",
                4: "ROUTING_FAIL_TIMEOUT",
                5: "ROUTING_INVALID",
                6: "ROUTING_INFEASIBLE"
            }

            print("Routing status meaning:", status_map.get(status, "UNKNOWN"))
            print("Returning parent solution.\n")

            # RETURN PARENT SOLUTION IMMEDIATELY
            fallback = ProblemState(current.parent_solution, current.data)
            fallback.unassigned = []
            return fallback

        print(f"[DAY {t}] ✅ Locked routes accepted")

        # -------------------------------------------------
        # Extract solution
        # -------------------------------------------------
        served = np.zeros(N)

        for v in range(V):
            route = []
            idx = routing.Start(v)

            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)

                if node != 0:
                    cust, q = nodes[node - 1]
                    route.append((cust, q))
                    repaired.solution["shipments"][t, v, cust] += q
                    served[cust] += q

                idx = sol.Value(routing.NextVar(idx))

            repaired.solution["routes"][t][v] = route
            print(f"[DAY {t}] Vehicle {v}: stops={len(route)}")

        print(f"[DAY {t}] Extraction complete")

    repaired.unassigned = []
    print("\n================ ORTOOLS LOCKED ROUTE REPAIR END =================")
    return repaired
"""


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
        init.objective(), 200, 0, num_iters
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
        eoh_experiment_file=f"U_C_N25_Q?_T?",
        lns_paras = {
            "num_iters" : 1000,
            "runtime_limit" : 10,
            "no_improvement_time" : 10,
            "no_improvement_iters" : 100,
        }
    )

    lns.init_lns(8, paras= paras_instance)

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
