import time

from dotenv import load_dotenv

from solver.ceoh.utils.getParas import Paras

from libraries.lns.base_class import LNS_Base

import numpy.random as rnd

import os
import json

from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import NoImprovement

from libraries.lns.core.StopCriteria import MaxIterationsAndMaxRuntime, OrStop, NoImprovementTime
READ_FROM_JSON = False


PROBLEM = "lns_IRP_inventory"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def routing_cost_from_routes(routes, data):

    C = data["cost_matrix"]
    total_cost = 0.0

    for t_routes in routes.values():
        for route in t_routes.values():

            if not route:
                continue

            prev = 0  # depot

            for i, _ in route:
                node = i + 1
                total_cost += C[prev][node]
                prev = node

            total_cost += C[prev][0]  # return to depot

    return total_cost

def inventory_cost_from_shipments(shipments, data):

    T, V, N = shipments.shape

    depot_I = float(data["depot_I0"])
    cust_I = data["cust_I0"].astype(float).copy()

    depot_prod = data["depot_prod"]
    demand = data["cust_demand"]
    depot_cost = data["depot_inv_cost"]
    cust_cost = data["cust_inv_cost"]

    # PRECOMPUTE ONCE
    period_outflow = shipments.sum(axis=(1, 2))   # shape (T,)
    cust_inflow = shipments.sum(axis=1)           # shape (T, N)

    cost = 0.0

    for t in range(T):

        depot_I += depot_prod[t]
        depot_I -= period_outflow[t]

        cust_I += cust_inflow[t]
        cust_I -= demand[:, t]

        cost += depot_I * depot_cost
        cost += (cust_I * cust_cost).sum()

    return cost

class ProblemState:
    def __init__(self, solution, data):
        self.solution = solution   # {"shipments": ..., "routes": ...}
        self.data = data
        self.unassigned = []
        self.parent_solution = copy.deepcopy(solution)

    def get_context(self):
        shipments = self.solution["shipments"]
        T, V, N = shipments.shape
        used = int(np.count_nonzero(shipments))
        return [f"shipments: {used} nonzeros"]

    def objective(self) -> float:
        total_costs = self.inventory_cost() + self.routing_cost()
        return total_costs

    def inventory_cost(self) -> float:
        return inventory_cost_from_shipments(self.solution["shipments"], self.data)

    def routing_cost(self):
        return routing_cost_from_routes(self.solution["routes"], self.data)

    def __str__(self):
        return f"Solution: {self.objective()} \n with routes: {self.solution}"


from problems.inventory_routing.get_instance import load_full_skalnes_instances

from problems.inventory_routing.get_initial_solution import load_initial_construction_solution_with_routes

import json
import numpy as np

def load_solution_from_json(filepath):
    """
    Loads a solution written by write_state_to_json and reconstructs
    shipments (np.ndarray) and routes (dict[t][v]).
    """
    import json
    import numpy as np
    import os

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Solution file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    sol_json = data["solution"]

    # --- Shipments ---
    shipments = np.array(sol_json["shipments"], dtype=float)

    # --- Routes ---
    routes = {
        int(t): {
            int(v): [(int(i), float(q)) for i, q in sol_json["routes"][t][v]]
            for v in sol_json["routes"][t]
        }
        for t in sol_json["routes"]
    }

    return {
        "shipments": shipments,
        "routes": routes
    }

def initial_state(data, paras, instance_seed: int) -> ProblemState:

    instances = load_full_skalnes_instances(paras.eoh_experiment_file)
    instance = instances[instance_seed % len(instances)]

    print(instance["meta"]["file"])

    sol = load_initial_construction_solution_with_routes(instance["meta"]["file"])

    #print("### INITIAL SOLUTION ###")
    #print_solution(ProblemState(sol, data))

    if READ_FROM_JSON:
        # ---- Load from JSON if provided ----
        json_path = f"solution_seed_{instance_seed}.json"

        if json_path is not None:
            sol = load_solution_from_json(json_path)
            print(f"Loaded initial solution from JSON: {json_path}")

    return ProblemState(sol, data)

####################################################################
# Repair (Random)

import copy
import numpy as np

def apply_scoring(state: ProblemState, scores, k):
    """Apply destroy scores by zeroing out the top-k scored shipment entries.

        Scores are expected to be a 3D nested structure aligned with shipments:
            scores[t][v][i] = score (higher means more likely to be destroyed)

        The function ranks all scored entries and sets the corresponding
        `shipments[t, v, i]` to 0.0 for the top `k` entries.

        Parameters
        ----------
        state : ProblemState
            State whose shipment tensor will be modified in-place.
        scores : list
            Nested list of scores with shape (T, V, N). Entries may be numeric.
        k : int
            Number of shipment entries to destroy (set to zero).

        Returns
        -------
        ProblemState
            The same `state` object, with updated shipments.
    """
    shipments = state.solution["shipments"]

    ranked = []

    for t in range(len(scores)):
        for v in range(len(scores[t])):
            for i in range(len(scores[t][v])):
                s = scores[t][v][i]
                if s is not None:
                    ranked.append((s, t, v, i))

    ranked.sort(reverse=True)

    for _, t, v, i in ranked[:k]:
        shipments[t, v, i] = 0.0

    state.solution["shipments"] = shipments
    return state


def destroy(current: ProblemState, rng: rnd.Generator) -> ProblemState:
    destroyed = clone_state(current)
    scores,k = destroy_scoring(copy.copy(destroyed.solution), copy.copy(destroyed.data), rng)
    #scores,k = destroy_scoring(destroyed.solution, destroyed.data, rng)
    destroyed = apply_scoring(destroyed, scores, k)

    return destroyed

def destroy_scoring(solution, data, rng):
    shipments = solution["shipments"]
    T, V, N = shipments.shape

    scores = []   # scores[t][v][i] = random score or None

    active_count = 0

    for t in range(T):
        scores_t = []
        for v in range(V):
            scores_v = []
            for i in range(N):
                if shipments[t, v, i] > 1e-9:
                    scores_v.append(float(rng.random()))
                    active_count += 1
                else:
                    scores_v.append(0)
            scores_t.append(scores_v)
        scores.append(scores_t)

    if active_count == 0:
        return scores, 0

    low = max(1, int(0.05 * active_count))
    high = max(low, int(0.20 * active_count))
    k = int(rng.integers(low, high + 1))

    return scores, k


####################################################################
# Repair
####################################################################
def rebuild_routes_from_shipments(shipments, routes, data, rng):
    C = data["cost_matrix"]
    T, V, N = shipments.shape

    for t in range(T):
        for v in range(V):

            route = routes[t][v]

            # ---- 1. Build live quantity table ----
            route_qty = {}
            for cust, q in route:
                route_qty[cust] = route_qty.get(cust, 0.0) + q

            # ---- 2. Synchronize each customer one by one ----
            for i in range(N):
                target = float(shipments[t, v, i])
                current = route_qty.get(i, 0.0)
                diff = target - current

                # Insert missing quantity
                if diff > 1e-9:
                    if i in route_qty:
                        # --- Update existing stop quantity ---
                        for p, (cust, q) in enumerate(route):
                            if cust == i:
                                route[p] = (cust, q + diff)
                                break
                    else:
                        # --- Insert new stop only if customer not present ---
                        best_pos = 0
                        best_delta = float("inf")

                        for p in range(len(route) + 1):
                            a = 0 if p == 0 else route[p - 1][0] + 1
                            b = 0 if p == len(route) else route[p][0] + 1
                            delta = C[a][i + 1] + C[i + 1][b] - C[a][b]
                            if delta < best_delta:
                                best_delta = delta
                                best_pos = p

                        route.insert(best_pos, (i, diff))

                    route_qty[i] = target


                # Remove excess quantity
                elif diff < -1e-9:
                    remaining = -diff
                    new_route = []

                    for cust, q in route:
                        if cust == i and remaining > 1e-9:
                            if q <= remaining + 1e-9:
                                remaining -= q
                            else:
                                new_route.append((cust, q - remaining))
                                remaining = 0.0
                        else:
                            new_route.append((cust, q))

                    route[:] = new_route
                    route_qty[i] = target

            # ---- 3. Final hard check (debug safety) ----
            ship_sum = float(shipments[t, v].sum())
            route_sum = sum(q for _, q in route)

            if abs(ship_sum - route_sum) > 1e-6:
                raise RuntimeError(
                    f"Route sync failed at t={t}, v={v}: "
                    f"shipments={ship_sum}, route={route_sum}"
                )

    return routes


def repair(destroyed: ProblemState, rng: rnd.Generator) -> ProblemState:
    state = clone_state(destroyed)
    shipments = state.solution["shipments"]
    data = state.data

    # Ensure numpy arrays
    depot_prod = np.asarray(data["depot_prod"], dtype=float)
    demand = np.asarray(data["cust_demand"], dtype=float)  # shape (N, T)
    cust_I0 = np.asarray(data["cust_I0"], dtype=float)
    cust_Lmax = np.asarray(data["cust_Lmax"], dtype=float)

    T, V, N = shipments.shape
    veh_cap = float(data["veh_cap"])
    depot_I0 = float(data["depot_I0"])
    eps = 1e-9

    # -----------------------------
    # Precompute aggregates ONCE
    # -----------------------------
    # vehicle loads
    vehicle_loads = shipments.sum(axis=2)  # (T,V)
    # total shipments per period
    period_outflow = vehicle_loads.sum(axis=1)  # (T,)
    # customer inflow per period
    cust_inflow = shipments.sum(axis=1)  # (T,N)

    # depot end inventory per period
    depot_end = depot_I0 + np.cumsum(depot_prod - period_outflow)  # (T,)

    # customer inventory trajectory (end-of-period)
    # demand is (N,T) -> transpose to (T,N)
    inv = cust_I0[None, :] + np.cumsum(cust_inflow - demand.T, axis=0)  # (T,N)

    # -----------------------------
    # Repair loop
    # -----------------------------
    while True:
        shortage_mask = inv < -eps
        if not shortage_mask.any():
            break

        # earliest period with shortage
        t = int(np.where(shortage_mask.any(axis=1))[0][0])

        # largest shortage in that period
        i = int(np.argmin(inv[t]))
        need = -float(inv[t, i])
        remaining = need

        # allowed periods: 0..t
        col = shipments[: t + 1, :, i]  # shape (t+1, V)

        K1 = np.argwhere(col > eps)
        K3 = np.argwhere(col <= eps)

        if len(K1) > 1:
            rng.shuffle(K1)
        if len(K3) > 1:
            rng.shuffle(K3)

        for tau_v in np.vstack([K1, K3]):
            if remaining <= eps:
                break

            tau = int(tau_v[0])
            v = int(tau_v[1])

            # ---- Vehicle capacity ----
            free_cap = veh_cap - vehicle_loads[tau, v]
            if free_cap <= eps:
                continue

            # ---- Depot inventory feasibility ----
            depot_tau = depot_end[tau]
            if depot_tau <= eps:
                continue

            # ---- Customer Lmax feasibility ----
            # Must ensure for ALL s >= tau:
            # inv[s,i] + ship <= Lmax
            max_ship = float("inf")
            for s in range(tau, T):
                slack = cust_Lmax[i] - inv[s, i]
                if slack < max_ship:
                    max_ship = slack

            if max_ship <= eps:
                continue

            ship = min(free_cap, remaining, depot_tau, max_ship)
            if ship <= eps:
                continue

            # -------------------------
            # Apply shipment
            # -------------------------
            shipments[tau, v, i] += ship
            vehicle_loads[tau, v] += ship
            period_outflow[tau] += ship
            cust_inflow[tau, i] += ship

            depot_end[tau:] -= ship
            inv[tau:, i] += ship

            remaining -= ship

        # If we cannot fix the shortage
        if remaining > 1e-6:
            emergency_repaired = ProblemState(destroyed.parent_solution, destroyed.data)
            print("Return parent because unable to repair INVENTORY.")
            return emergency_repaired

    # -----------------------------
    # Rebuild routes for consistency
    # -----------------------------
    state.solution["shipments"] = shipments
    state.solution["routes"] = rebuild_routes_from_shipments(
        shipments, state.solution["routes"], data, rng
    )
    return state


####################################################################
# Utils
def load_data(instance_seed=0, paras=None):
    instances = load_full_skalnes_instances(paras.eoh_experiment_file)
    instance = instances[instance_seed % len(instances)]

    T = instance["meta"]["nPeriods"]
    V = instance["vehicles"]["fleet_size"]
    N = len(instance["customers"])

    data = {
        #"instance": instance,
        "T": T,
        "V": V,
        "N": N,
        "veh_cap": instance["vehicles"]["capacity"],
        "depot_inv_cost": instance["depot"]["inventory_cost"],
        "depot_prod": np.array(instance["depot"]["production"], float),
        "depot_I0": instance["depot"]["initial_inventory"],
        "cust_inv_cost": np.array([c["inventory_cost"] for c in instance["customers"]], float),
        "cust_I0": np.array([c["initial_inventory"] for c in instance["customers"]], float),
        "cust_Lmax": np.array([c["max_inventory"] for c in instance["customers"]], float),
        "cust_demand": np.array([c["demand"] for c in instance["customers"]], float),
        "cost_matrix": instance["cost_matrix"]
    }
    return data

def clone_state(state: ProblemState) -> ProblemState:
    """
    Efficient deep copy of ProblemState.

    Copies only mutable components:
        - shipments (numpy array)
        - routes (nested dict of lists)

    Shares:
        - data (assumed read-only)
    """

    # Fast C-level memory copy (O(n) contiguous copy)
    shipments_copy = state.solution["shipments"].copy()

    # Routes are nested dict[t][v] -> list[(i,q)]
    # Lists must be copied to avoid aliasing
    routes_copy = {
        t: {
            v: route.copy()   # shallow copy of list is enough (tuples are immutable)
            for v, route in vehicles.items()
        }
        for t, vehicles in state.solution["routes"].items()
    }

    # Construct new state
    new_state = ProblemState(
        solution={
            "shipments": shipments_copy,
            "routes": routes_copy,
        },
        data=state.data  # safe to share
    )

    # Optional: copy parent_solution reference if you rely on it
    new_state.parent_solution = state.parent_solution

    return new_state

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
    shipments = state.solution["shipments"]
    routes = state.solution["routes"]
    data = state.data

    T, V, N = shipments.shape

    depot_I = float(data["depot_I0"])
    cust_I = data["cust_I0"].astype(float).copy()

    inv_cost = state.inventory_cost()
    route_cost = state.routing_cost()
    total_cost = inv_cost + route_cost

    for t in range(T):
        print(f"Period {t}")

        # ---- Production arrives ----
        depot_I += float(data["depot_prod"][t])

        # ---- Routes ----
        for v in range(V):
            route = routes[t][v]
            if not route:
                continue
            load = sum(q for _, q in route)
            stops = " --> ".join(f"C{i}({int(q)})" for i, q in route)
            print(f"  V{v} [{int(load)}/{int(data['veh_cap'])}] : {stops}")

        # ---- Shipments leave depot ----
        depot_I -= float(shipments[t].sum())

        # ---- Inventory update ----
        for i in range(N):
            cust_I[i] += float(shipments[t, :, i].sum())
            cust_I[i] -= float(data["cust_demand"][i, t])

        # ---- Inventory summary ----
        cust_line = "  ".join(f"C{i}:{int(cust_I[i])}" for i in range(N))
        print(f"  Inventory --> Depot:{int(depot_I)} | {cust_line}")

        print("\n==================== SOLUTION ====================")
        print(f"Objective : {int(total_cost):,}")
        print(f"  Inventory : {int(inv_cost):,}")
        print(f"  Routing   : {int(route_cost):,}")
        print("================================================\n")


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
        eoh_experiment_file=f"U_C_N25_Q18_T?",
        lns_paras = {
            "num_iters" : 1000,
            "runtime_limit" : 10,
            "no_improvement_time" : 10,
            "no_improvement_iters" : 1000,
        }
    )

    lns.init_lns(0, paras= paras_instance)

    lns.run()
    best_state = lns.get_result()
    statistics = lns.get_statistics_summary()
    print(best_state.objective())
    print("initial_objective ",statistics["initial_objective"])
    print("best_objective ",statistics["best_objective"])
    lns.plot_objective()
    result = lns.get_result()
    #print(lns.get_statistics_summary())
    print("time passed", time.time() - start)

    best_state = lns.get_result()

    print("\n\n================ FINAL SOLUTION =================")
    print_solution(best_state)


    print("\n\n=================== OPTIMIZATION SUMMARY ===================")
    print("Initial Objective :", statistics["initial_objective"])
    print("Best Objective    :", statistics["best_objective"])
    print("Time Passed       :", time.time() - start)
