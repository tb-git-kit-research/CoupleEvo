import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

import os
import subprocess
from copy import deepcopy
import time

from dotenv import load_dotenv

from libraries.lns.base_class import LNS_Base
from libraries.lns.wrapper.lns_IRP_inventory import ProblemState as ProblemState_problem_one
from libraries.lns.wrapper.lns_IRP_vrp import ProblemState as ProblemStateRoute
from problems.inventory_routing.get_instance import load_full_skalnes_instances

from solver.ceoh.utils.getParas import Paras
import numpy as np

PLOTTING = False
PRINTOUT = False
SAVE_SOLUTION = False

def check_solution(state, label=""):
    sol = state.solution
    data = state.data

    shipments = sol["shipments"]
    routes = sol["routes"]

    T, V, N = shipments.shape
    cap = float(data["veh_cap"])

    # ---- 1. Capacity & shipment–route consistency ----
    for t in range(T):
        for v in range(V):
            ship_load = float(shipments[t, v].sum())
            route_load = sum(q for _, q in routes[t][v])

            if abs(ship_load - route_load) > 1e-6:
                raise RuntimeError(
                    f"[{label}] Shipment–Route mismatch at t={t}, v={v}: "
                    f"shipments={ship_load}, route={route_load}"
                )

            if ship_load > cap + 1e-6:
                raise RuntimeError(
                    f"[{label}] Capacity violation at t={t}, v={v}: {ship_load} > {cap}"
                )

    # ---- 2. Inventory feasibility ----
    depot_I = float(data["depot_I0"])
    cust_I = data["cust_I0"].astype(float).copy()

    for t in range(T):
        depot_I += float(data["depot_prod"][t])
        depot_I -= float(shipments[t].sum())

        if depot_I < -1e-6:
            raise RuntimeError(f"[{label}] Depot stockout at period {t}: {depot_I}")

        for i in range(N):
            cust_I[i] += float(shipments[t, :, i].sum())
            cust_I[i] -= float(data["cust_demand"][i, t])

            if cust_I[i] < -1e-6:
                raise RuntimeError(
                    f"[{label}] Customer {i} stockout at period {t}: {cust_I[i]}"
                )

            if cust_I[i] > data['cust_Lmax'][i] + 1e-6:
                raise RuntimeError(
                    f"[{label}] Customer {i} overflow at period {t}: {cust_I[i]} | {data['cust_Lmax'][i]}"
                )

    return True


def run_alternating_alns(cycles: int = 100,
                         paras_problem_one=None,
                         paras_route=None,
                         instance_seed=0,
                         lns_problem_one=None,
                         lns_route=None,
                         eoh_experiment_file=None,
                         timeout=60):

    if lns_problem_one is None:
        lns_problem_one = LNS_Base(problem="lns_IRP_inventory")
    lns_problem_one.init_lns(instance_seed, paras=paras_problem_one)
    current_state = lns_problem_one._init_sol

    #check_solution(current_state, label=f"Initial Inventory solution.")

    if lns_route is None:
        lns_route = LNS_Base(problem="lns_IRP_vrp")
    lns_route.init_lns(instance_seed, paras=paras_route)

    lns_route._init_sol = ProblemStateRoute(
        solution=current_state.solution,
        data=current_state.data
    )

    #check_solution(lns_route._init_sol, label=f"Initial VRP solution.")

    objective_history = []
    inventory_history = []
    routing_history = []

    def record(state):
        inv = state.inventory_cost()
        rout = state.routing_cost()
        inventory_history.append(inv)
        routing_history.append(rout)
        objective_history.append(inv + rout)

    record(lns_route._init_sol)

    route_times = []
    problem_one_times = []
    cycle_times = []

    overall_time = 0
    for cycle in range(cycles):
        print(cycle)

        cycle_start = time.time()

        # -------- Inventory Phase --------
        problem_one_time_start = time.time()

        lns_problem_one._init_sol = ProblemState_problem_one(
            solution=current_state.solution,
            data=current_state.data
        )

        lns_problem_one.init_lns(instance_seed, paras=paras_problem_one)
        lns_problem_one.run()
        current_state = lns_problem_one._result.best_state

        #check_solution(current_state, label=f"After Inventory LNS (cycle {cycle})")

        problem_one_times.append(time.time() - problem_one_time_start)

        if overall_time + time.time() - cycle_start > timeout:
            if PRINTOUT:
                print("Timout in LNS Break cycle")
            break

        # -------- Routing Phase --------
        route_time_start = time.time()

        lns_route._init_sol = ProblemStateRoute(
            solution=current_state.solution,
            data=current_state.data,
        )

        lns_route.init_lns(instance_seed, paras=paras_route)
        lns_route.run()
        current_state = lns_route._result.best_state

        #check_solution(current_state, label=f"After Inventory LNS (cycle {cycle})")

        record(current_state)

        route_times.append(time.time() - route_time_start)
        cycle_times.append(time.time() - cycle_start)

        overall_time += cycle_times[-1]
        if overall_time > timeout:
            if PRINTOUT:
                print("Timout in LNS Break cycle")
            break
        print(objective_history[-1])

    print(f"Initial Objective: {objective_history[0]:.2f}")
    print(f"Final Objective: {objective_history[-1]:.2f}")
    if PLOTTING:
        plot_objective_graph(
            eoh_experiment_file,
            instance_seed,
            objective_history,
            inventory_history,
            routing_history
        )

    if PLOTTING:
        plot_daily_routes_and_customers(current_state, eoh_experiment_file, instance_seed)

    if PRINTOUT:
        print_routes_and_inventory(current_state)
        print(current_state.solution["shipments"])
        print(current_state.solution["routes"])

    if SAVE_SOLUTION:
        write_state_to_json(current_state, f"solution_seed_{instance_seed}.json")

    return {
        "objective_history": objective_history,
        "inventory_history": inventory_history,
        "routing_history": routing_history,
        "cycle_times": cycle_times,
    }

def write_state_to_json(state, filepath):
    """
    Writes the full solution (shipments, routes, costs) to a JSON file.
    """
    data = {
        "objective": state.objective(),
        "inventory_cost": state.inventory_cost(),
        "routing_cost": state.routing_cost(),
        "solution": {
            "shipments": state.solution["shipments"],
            "routes": {
                str(t): {
                    str(v): [(int(i), float(q)) for i, q in state.solution["routes"][t][v]]
                    for v in state.solution["routes"][t]
                }
                for t in state.solution["routes"]
            }
        }
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    print(f"Solution written to {filepath}")

def plot_objective_graph(eoh_experiment_file, instance_seed,
                         objective_history, inventory_history, routing_history):
    instance = load_full_skalnes_instances(eoh_experiment_file)[instance_seed]

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Left axis: Objective + Routing ---
    ax1.plot(objective_history, label="Total Objective", linewidth=2)
    ax1.plot(routing_history, label="Routing Cost", linestyle=":")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Objective / Routing Cost")
    ax1.grid(True)

    # --- Right axis: Inventory ---
    ax2 = ax1.twinx()
    ax2.plot(inventory_history, label="Inventory Cost", linestyle="--")
    ax2.set_ylabel("Inventory Cost")

    # --- Title ---
    ax1.set_title(f"{instance['meta']['file']} | seed {instance_seed}")

    # --- Unified Legend ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.tight_layout()
    plt.show()

def print_routes_and_inventory(state):
    """
    Prints vehicle routes and inventory evolution period by period.
    """
    sol = state.solution
    data = state.data

    shipments = sol["shipments"]
    routes = sol["routes"]

    T, V, N = shipments.shape

    depot_I = float(data["depot_I0"])
    cust_I = data["cust_I0"].astype(float).copy()

    print("\n================ ROUTES & INVENTORY =================")

    for t in range(T):
        print(f"\n--- Period {t} ---")

        # Production arrives
        depot_I += float(data["depot_prod"][t])

        # Routes
        for v in range(V):
            route = routes[t][v]
            if not route:
                continue

            load = sum(q for _, q in route)
            stops = " -> ".join(f"C{i}({int(q)})" for i, q in route)
            print(f"Vehicle {v} [{int(load)}/{int(data['veh_cap'])}]: {stops}")

        # Shipments leave depot
        depot_I -= float(shipments[t].sum())

        # Inventory update
        for i in range(N):
            cust_I[i] += float(shipments[t, :, i].sum())
            cust_I[i] -= float(data["cust_demand"][i, t])

        # Inventory status
        cust_str = " | ".join(f"C{i}: {int(cust_I[i])}" for i in range(N))
        print(f"Depot inventory: {int(depot_I)}")
        print(f"Customer inventory: {cust_str}")

    print("\n================ COST SUMMARY =================")
    print(f"Inventory cost : {state.inventory_cost():.2f}")
    print(f"Routing cost   : {state.routing_cost():.2f}")
    print(f"Total cost     : {state.objective():.2f}")
    print("================================================\n")

def plot_daily_routes_and_customers(state, eoh_experiment_file, instance_seed):
    """
    Creates one subplot per day showing:
    - vehicle routes
    - active customers (served that day)
    - inactive customers (transparent)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    sol = state.solution
    data = state.data
    instance = load_full_skalnes_instances(eoh_experiment_file)[instance_seed]
    #instance = data["instance"]

    routes = sol["routes"]
    shipments = sol["shipments"]

    T, V, N = shipments.shape

    # --- Coordinates ---
    depot_x = instance["depot"]["x"]
    depot_y = instance["depot"]["y"]

    cust_x = np.array([c["x"] for c in instance["customers"]])
    cust_y = np.array([c["y"] for c in instance["customers"]])

    # --- Figure layout ---
    fig, axes = plt.subplots(
        nrows=1,
        ncols=T,
        figsize=(5 * T, 5),
        squeeze=False
    )

    for t in range(T):
        ax = axes[0, t]
        ax.set_title(f"Day {t}")
        ax.set_aspect("equal")

        # --- Active customers ---
        active = np.where(shipments[t].sum(axis=0) > 1e-9)[0]
        inactive = np.where(shipments[t].sum(axis=0) <= 1e-9)[0]

        # --- Plot inactive customers ---
        ax.scatter(
            cust_x[inactive],
            cust_y[inactive],
            s=40,
            alpha=0.2,
            label="Inactive"
        )

        # --- Plot active customers ---
        ax.scatter(
            cust_x[active],
            cust_y[active],
            s=60,
            alpha=0.9,
            label="Active"
        )

        # --- Plot depot ---
        ax.scatter(
            depot_x,
            depot_y,
            s=120,
            marker="s",
            label="Depot"
        )

        # --- Draw routes ---
        for v in range(V):
            route = routes[t][v]
            if not route:
                continue

            path_x = [depot_x]
            path_y = [depot_y]

            for i, _ in route:
                path_x.append(cust_x[i])
                path_y.append(cust_y[i])

            path_x.append(depot_x)
            path_y.append(depot_y)

            ax.plot(path_x, path_y, linewidth=1.5, alpha=0.8)

        ax.legend(loc="best", fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":


    subprocess.run(["pip", "install", "."], stdout=subprocess.DEVNULL)

    load_dotenv()
    eoh_experiment_file = f"U_C_N25_Q18_T?"

    results_list = []

    seeds = range(len(load_full_skalnes_instances(eoh_experiment_file)))
    for instance_seed in seeds:
        paras_problem_one = Paras()
        # add eoh_experiment_file to paras
        paras_problem_one.set_paras(
            eoh_experiment_file=eoh_experiment_file,
            lns_paras = {
                "num_iters": 1000,
                "runtime_limit": 10,
                "no_improvement_time": 10,
                "no_improvement_iters": 1000,
            }
        )

        paras_route = Paras()
        # add eoh_experiment_file to paras
        paras_route.set_paras(
            eoh_experiment_file=eoh_experiment_file,
            lns_paras = {
                "num_iters": 1000,
                "runtime_limit": 10,
                "no_improvement_time": 10,
                "no_improvement_iters": 1000,
            }
        )
        results_dict = run_alternating_alns(cycles=20,
                                            paras_problem_one= paras_problem_one,
                                            paras_route = paras_route,
                                            instance_seed= instance_seed,
                                            eoh_experiment_file = eoh_experiment_file,
                                            timeout = 60)
        results_list.append(results_dict)

    # extract final objective value from each run
    final_vals = [run["objective_history"][-1] for run in results_list]

    mean_fitness = np.mean(final_vals)
    std_fitness = np.std(final_vals)

    print("Final objectives:", final_vals)
    print("Mean fitness:", mean_fitness)
    print("Std:", std_fitness)

