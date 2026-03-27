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
    """
    Computes total routing cost from explicit route representation.

    routes[t][v] = [(i1, q1), (i2, q2), ...]
    0 is the depot in cost_matrix
    customers are indexed 0..N-1 in shipments and 1..N in cost_matrix
    """

    C = data["cost_matrix"]
    total_cost = 0.0

    for t in routes:
        for v in routes[t]:
            route = routes[t][v]
            if not route:
                continue

            # Construct full tour: depot -> customers -> depot
            path = [0]
            for i, _ in route:
                path.append(i + 1)
            path.append(0)

            # Sum arc costs
            for k in range(len(path) - 1):
                total_cost += C[path[k]][path[k + 1]]

    return total_cost

def inventory_cost_from_shipments(shipments, data):
    sol = shipments

    T, V, N = sol.shape

    depot_I = data["depot_I0"]
    cust_I = data["cust_I0"].copy()

    cost = 0.0

    # Initial Inventory costs
    #cost += depot_I * data["depot_inv_cost"]
    #for i in range(N):
    #    cost += cust_I[i] * data["cust_inv_cost"][i]

    for t in range(T):
        t_cost = 0
        # Production arrives
        depot_I += data["depot_prod"][t]

        # Shipments leave depot
        depot_I -= sol[t].sum()

        # Customer inventory update
        for i in range(N):
            cust_I[i] += sol[t, :, i].sum()
            cust_I[i] -= data["cust_demand"][i, t]

        # Holding cost
        t_cost += depot_I * data["depot_inv_cost"]
        for i in range(N):
            t_cost += cust_I[i] * data["cust_inv_cost"][i]
        cost += t_cost
    return cost

def convert_greedy_solution(greedy, data, id_to_index):
    import numpy as np

    T, V, N = data["T"], data["V"], data["N"]
    cap = float(data["veh_cap"])

    shipments = np.zeros((T, V, N))
    routes = {t: {v: [] for v in range(V)} for t in range(T)}

    for p in greedy["periods"]:
        t = p["t"]
        deliveries = p["deliveries"]

        # vehicle loads for this period
        load = [0.0] * V

        # Sort deliveries by descending quantity (improves initial routing quality)
        items = sorted(deliveries.items(), key=lambda x: -x[1])

        for cust_id, qty in items:
            if qty <= 1e-9:
                continue

            i = id_to_index[cust_id]
            remaining = qty

            while remaining > 1e-9:
                # choose vehicle with most remaining capacity
                v = max(range(V), key=lambda k: cap - load[k])
                space = cap - load[v]

                if space <= 1e-9:
                    raise RuntimeError(f"Infeasible greedy solution at period {t}")

                ship = min(space, remaining)

                shipments[t, v, i] += ship
                routes[t][v].append((i, ship))

                load[v] += ship
                remaining -= ship

    return {
        "shipments": shipments,
        "routes": routes
    }

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

def initial_state(data, paras, instance_seed: int) -> ProblemState:

    instances = load_full_skalnes_instances(paras.eoh_experiment_file)
    instance = instances[instance_seed % len(instances)]

    print(instance["meta"]["file"])

    sol = load_initial_construction_solution_with_routes(instance["meta"]["file"], paras.eoh_experiment_file)

    #print("### INITIAL SOLUTION ###")
    #print_solution(ProblemState(sol, data))

    return ProblemState(sol, data)

####################################################################
# Repair (Random)

import copy
import numpy as np

def apply_scoring(state: ProblemState, scores, k):
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
    destroyed = copy.deepcopy(current)
    scores,k = destroy_scoring(destroyed.solution, destroyed.data, rng)
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
                    best_pos = 0
                    best_delta = float("inf")

                    for p in range(len(route) + 1):
                        a = 0 if p == 0 else route[p-1][0] + 1
                        b = 0 if p == len(route) else route[p][0] + 1
                        delta = C[a][i+1] + C[i+1][b] - C[a][b]
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = p

                    route.insert(best_pos, (i, diff))
                    route_qty[i] = route_qty.get(i, 0.0) + diff

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
    import copy

    state = copy.deepcopy(destroyed)

    shipments = state.solution["shipments"]
    data = state.data

    T, V, N = shipments.shape
    veh_cap = float(data["veh_cap"])

    depot_I0 = float(data["depot_I0"])
    cust_I0 = data["cust_I0"].astype(float)

    cust_Lmax = data["cust_Lmax"].astype(float)

    while True:
        # --- Recompute inventories from scratch ---
        depot_I = depot_I0
        cust_I = cust_I0.copy()
        shortages = []

        for t in range(T):
            depot_I += float(data["depot_prod"][t])
            depot_I -= float(shipments[t].sum())

            for i in range(N):
                cust_I[i] += float(shipments[t, :, i].sum())
                cust_I[i] -= float(data["cust_demand"][i, t])

            for i in range(N):
                if cust_I[i] < -1e-9:
                    shortages.append((t, i, -cust_I[i]))

        if not shortages:
            break

        # --- Select earliest period, largest shortage in that period ---
        t_min = min(s[0] for s in shortages)
        t, i, need = max((s for s in shortages if s[0] == t_min), key=lambda x: x[2])

        remaining = need

        # --- Randomized period order for backfilling ---
        candidate_periods = list(range(t + 1))
        rng.shuffle(candidate_periods)

        for tau in candidate_periods:
            if remaining <= 1e-9:
                break

            for v in range(V):
                load = float(shipments[tau, v].sum())
                free_cap = veh_cap - load
                if free_cap <= 1e-9:
                    continue

                # Depot inventory available up to tau
                depot_tau = depot_I0
                for tt in range(tau + 1):
                    depot_tau += float(data["depot_prod"][tt])
                    depot_tau -= float(shipments[tt].sum())

                # Compute maximum feasible ship without creating future overflow
                max_ship = float("inf")

                cust_temp = cust_I0[i]
                for s in range(T):
                    cust_temp += float(shipments[s, :, i].sum())
                    cust_temp -= float(data["cust_demand"][i, s])

                    if s >= tau:
                        max_ship = min(max_ship, cust_Lmax[i] - cust_temp)

                if max_ship <= 1e-9:
                    continue

                ship = min(free_cap, remaining, depot_tau, max_ship)

                if ship <= 1e-9:
                    continue

                shipments[tau, v, i] += ship
                remaining -= ship

                if remaining <= 1e-9:
                    break

        # If we cannot fix the shortage, stop (instance or current state infeasible)
        if remaining > 1e-6:
            emergingy_repaired = ProblemState(destroyed.parent_solution, destroyed.data)
            print("Return parent because unable to repair INVENTORY. ")
            return emergingy_repaired

    # Commit updated shipments back
    state.solution["shipments"] = shipments

    # IMPORTANT: routes are derived; rebuild them so routing_cost/printing are consistent
    state.solution["routes"] = rebuild_routes_from_shipments(shipments, destroyed.solution["routes"], data, rng)

    return state

def check_inventory_outages(state: ProblemState):
    sol = state.solution
    data = state.data

    T, V, N = sol.shape

    depot_I = float(data["depot_I0"])
    cust_I = data["cust_I0"].astype(float).copy()

    outages = []

    for t in range(T):
        depot_I += float(data["depot_prod"][t])
        depot_I -= float(sol[t].sum())

        for i in range(N):
            cust_I[i] += float(sol[t, :, i].sum())
            cust_I[i] -= float(data["cust_demand"][i, t])

            if cust_I[i] < -1e-9:
                outages.append({
                    "period": t,
                    "customer": i,
                    "inventory": cust_I[i]
                })

    return outages


####################################################################
# Utils
def load_data(instance_seed=0, paras=None):
    instances = load_full_skalnes_instances(paras.eoh_experiment_file)
    instance = instances[instance_seed % len(instances)]

    T = instance["meta"]["nPeriods"]
    V = instance["vehicles"]["fleet_size"]
    N = len(instance["customers"])

    data = {
        "instance": instance,
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
        eoh_experiment_file=f"U_C_N25",
        lns_paras = {
            "num_iters" : 100,
            "runtime_limit" : 10,
            "no_improvement_time" : 10,
            "no_improvement_iters" : 100,
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
