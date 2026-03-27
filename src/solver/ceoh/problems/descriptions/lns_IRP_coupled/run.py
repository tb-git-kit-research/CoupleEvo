import inspect
import os
import time
import types
from multiprocessing import Process, Queue

import dotenv
import libraries.lns.wrapper.lns_IRP_vrp
import libraries.lns.wrapper.lns_IRP_inventory
import numpy as np

from problems.inventory_routing.get_instance import load_full_skalnes_instances

from libraries.lns.wrapper.run_alns_cycle_inventory_routing import run_alternating_alns

from solver.ceoh.problems.descriptions.lns_IRP_coupled.prompts import GetPrompts

from solver.ceoh.utils.get_coupled_heuristics import get_code_strings

from solver.ceoh.utils.getParas import Paras

from libraries.lns.base_class import LNS_Base
from problems.inventory_routing.get_HGS_objective import load_hgs_stats, load_all_hgs_stats

####################################################
# Config
INVENTORY_PROBLEM = 'lns_IRP_inventory'
VRP_PROBLEM = 'lns_IRP_vrp'
COUPLED_PROBLEM = 'lns_IRP_coupled'
SUBPROBLEMS = [INVENTORY_PROBLEM, VRP_PROBLEM]
TIMEOUT_SECONDS = 85
N_WORKERS = 10
####################################################

def _build_heuristic_module_inventory(code_string: str, problem: str) -> types.ModuleType:
    """Baut das Heuristik-Modul wie in deinem evaluate() zusammen."""
    imports = (
        "import copy\n"
        "import numpy as np\n"
        "import math\n"
        f"from libraries.lns.wrapper.{problem} import *\n\n"
    )
    destroy_wrapper = (
        "\n\n"
        "def destroy(current: ProblemState, rng: rnd.Generator) -> ProblemState:\n"
        "\tdestroyed = copy.deepcopy(current)\n"
        "\tscores,k = destroy_scoring(copy.deepcopy(destroyed.solution), copy.deepcopy(destroyed.data), rng)\n"
        "\tdestroyed = apply_scoring(destroyed, scores, k)\n"
        "\treturn destroyed \n\n"
    )
    full_code = imports + str(code_string) + destroy_wrapper
    mod = types.ModuleType("heuristic_module")
    exec(full_code, mod.__dict__)
    mod.code_string = full_code
    return mod

def _build_heuristic_module_vrp(code_string: str, problem: str) -> types.ModuleType:
    """Baut das Heuristik-Modul wie in deinem evaluate() zusammen."""
    imports = (
        "import copy\n"
        "import numpy as np\n"
        "import math\n"
        f"from libraries.lns.wrapper.{problem} import *\n\n"
    )
    destroy_wrapper = (
        "\n\n"
        "def destroy(current: ProblemState, rng: rnd.Generator) -> ProblemState:\n"
        "\tdestroyed = copy.deepcopy(current)\n"
        "\troutes = destroyed.solution['routes']\n"
        "\tshipments = destroyed.solution['shipments']\n"
        "\n"
        "\tfor t in routes:\n"
        "\t\tday_routes = routes[t]\n"
        "\t\tday_shipments = shipments[t]\n"
        "\n"
        "\t\tscores, k = destroy_scoring(day_routes, destroyed.data, rng)\n"
        "\n"
        "\t\tif k > 0:\n"
        "\t\t\tapply_scoring(day_routes, day_shipments, t, scores, k, destroyed)\n"
        "\n"
        "\treturn destroyed\n"
    )
    full_code = imports + str(code_string) + destroy_wrapper
    mod = types.ModuleType("heuristic_module")
    exec(full_code, mod.__dict__)
    mod.code_string = full_code
    return mod

def solve_seed_worker(task_queue, result_queue, code_string, paras):
    while True:
        seed = task_queue.get()
        if seed is None:
            break
        try:
            print(f"Evaluate heuritsic generated for {paras.target_subproblem}")

            # liste von lns wrappern
            lns_inventory = LNS_Base(INVENTORY_PROBLEM)
            lns_route = LNS_Base(VRP_PROBLEM)

            print(f"Heuritsic generated for {paras.target_subproblem}")

            inventory_code = None
            vrp_code = None

            if paras.target_subproblem == COUPLED_PROBLEM:
                # both heuristics passed explicitly
                if code_string is None:
                    raise ValueError("Coupled problem requires coupled_code_strings")

                inventory_code = code_string.get(INVENTORY_PROBLEM)
                vrp_code = code_string.get(VRP_PROBLEM)

            elif paras.target_subproblem == INVENTORY_PROBLEM:
                stored_code_strings = get_code_strings(paras)
                inventory_code = code_string
                vrp_code = stored_code_strings.get(VRP_PROBLEM)

            elif paras.target_subproblem == VRP_PROBLEM:
                stored_code_strings = get_code_strings(paras)
                vrp_code = code_string
                inventory_code = stored_code_strings.get(INVENTORY_PROBLEM)

            if inventory_code is not None:
                heuristic = _build_heuristic_module_inventory(inventory_code, INVENTORY_PROBLEM)
                lns_inventory.set_destroy(heuristic.destroy)
                print(f"Registered {INVENTORY_PROBLEM} destroy heuristic")

            if vrp_code is not None:
                heuristic = _build_heuristic_module_vrp(vrp_code, VRP_PROBLEM)
                lns_route.set_destroy(heuristic.destroy)
                print(f"Registered {VRP_PROBLEM} destroy heuristic")

            paras_problem_one = Paras()
            paras_problem_one.set_paras(
                eoh_experiment_file=paras.eoh_experiment_file,
                lns_paras=paras.lns_paras
            )

            paras_route = Paras()
            paras_route.set_paras(
                eoh_experiment_file=paras.eoh_experiment_file,
                lns_paras=paras.lns_paras_second
            )

            results_dict = run_alternating_alns(cycles=paras.alns_cycles_iters,
                                                paras_problem_one=paras_problem_one,
                                                paras_route=paras_route,
                                                instance_seed=seed,
                                                lns_problem_one=lns_inventory,
                                                lns_route=lns_route,
                                                eoh_experiment_file = paras.eoh_experiment_file,
                                                timeout = paras.alns_cycles_timeout)

            fitness = min(results_dict["objective_history"])

            result_queue.put({"seed": seed, "fitness": fitness, "results": results_dict})
        except Exception as e:
            result_queue.put({"seed": seed, "fitness": None, "error": str(e)})

class LNS_IRP_CONSTANT:
    def __init__(self, code_string=None, paras=None):

        self.paras = paras

        if paras.eoh_experiment_file:
            self.instances = load_full_skalnes_instances(paras.eoh_experiment_file)
            self.ref_solutions = load_all_hgs_stats(self.instances)
            self.seeds = list(range(len(self.instances)))
            global N_WORKERS
            N_WORKERS = min(len(self.seeds), N_WORKERS)


        self.prompts = GetPrompts()

        self.human_code_string = [inspect.getsource(libraries.lns.wrapper.lns_IRP_inventory.destroy_scoring),
                                  inspect.getsource(libraries.lns.wrapper.lns_IRP_vrp.destroy_scoring)]


    def get_subproblems(self):
        return SUBPROBLEMS

    def evaluate(self, code_string):
        seeds = self.seeds
        managers = []

        # Start Worker
        for _ in range(N_WORKERS):
            task_queue = Queue()
            result_queue = Queue()
            p = Process(target=solve_seed_worker,
                        args=(task_queue, result_queue, code_string,self.paras))
            p.start()
            managers.append((p, task_queue, result_queue))

        results = [None] * len(seeds)
        buckets = [[] for _ in range(N_WORKERS)]
        for i, seed in enumerate(seeds):
            buckets[i % N_WORKERS].append((i, seed))

        def run_manager(index, jobs):
            process, task_queue, result_queue = managers[index]
            for i, seed in jobs:
                task_queue.put(seed)
                start_time = time.time()
                res = None
                while time.time() - start_time < TIMEOUT_SECONDS:
                    try:
                        cand = result_queue.get(timeout=0.1)
                        res = cand
                        break
                    except:
                        pass
                results[i] = res
            task_queue.put(None)

            # Wait a little for graceful shutdown
            process.join(timeout=0.1)

            # If still alive (e.g., heuristic stuck in infinite loop), kill it
            if process.is_alive():
                print(f"[WARN] Worker {index} did not exit, force killing.")
                process.terminate()
                process.join()

        import threading
        threads = []
        for i in range(N_WORKERS):
            t = threading.Thread(target=run_manager, args=(i, buckets[i]))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        for r in results:
            print(r)

        fitness_values = [r["fitness"] for r in results]

        if any(f is None for f in fitness_values):
            raise ValueError("One or more fitness values are None")

        fitness_values = [
            float(r["fitness"])
            for r in results
            if r is not None and r.get("fitness") is not None
        ]

        if len(fitness_values) != len(self.seeds):
            raise ValueError("Missing fitness values for some seeds")


        reference_values = np.array(self.ref_solutions, dtype=float)
        fitness_values = np.array(fitness_values, dtype=float)

        # --- average fitness ---
        avg_objective = float(np.mean(fitness_values))

        # --- mean relative gap ---
        gaps = (fitness_values - reference_values) / reference_values
        mean_gap = float(np.mean(gaps))

        print("Fitness values:", fitness_values.tolist())
        print("Reference values:", reference_values.tolist())
        print("Durchschnitt Objective:", round(avg_objective, 4))
        print("Mean Gap:", round(mean_gap * 100, 2), "%\n")

        return mean_gap, results

if __name__ == "__main__":

    print(f"Start LNS for CYCLE {INVENTORY_PROBLEM} + {VRP_PROBLEM}")
    vrp_code = """  
def destroy_scoring(day_routes, data, rng):
    N = data['N']
    T = data['T']
    cost_matrix = data['cost_matrix']
    cust_demand = data['cust_demand']
    cust_I0 = data['cust_I0']
    cust_Lmax = data['cust_Lmax']
    cust_inv_cost = data['cust_inv_cost']
    
    # Calculate inventory risk exposure (how close to maximum capacity)
    inventory_risk = np.zeros(N)
    for i in range(N):
        if cust_Lmax[i] > 0:
            inventory_risk[i] = 1.0 - (cust_I0[i] / cust_Lmax[i])
    
    # Calculate demand-supply mismatch (average demand vs current inventory)
    demand_mean = np.mean(cust_demand, axis=1)
    supply_demand_mismatch = np.zeros(N)
    for i in range(N):
        if demand_mean[i] > 0:
            supply_demand_mismatch[i] = abs(cust_I0[i] - demand_mean[i]) / (demand_mean[i] + 1e-8)
    
    # Calculate route isolation (how many vehicles serve this customer)
    vehicle_count = np.zeros(N)
    for v, route in day_routes.items():
        if not route:
            continue
        for cust_idx, _ in route:
            vehicle_count[cust_idx] += 1
    
    # Calculate route efficiency impact (average travel cost when customer is removed)
    customer_route_impact = np.zeros(N)
    for v, route in day_routes.items():
        if not route:
            continue
        for i, (cust_idx, _) in enumerate(route):
            if len(route) <= 1:
                continue
            
            # Cost of removing customer from route
            cost_without = 0
            if i > 0:
                cost_without += cost_matrix[route[i-1][0] + 1][cust_idx + 1]
            if i < len(route) - 1:
                cost_without += cost_matrix[cust_idx + 1][route[i+1][0] + 1]
            
            # Cost of connecting neighbors
            cost_with = 0
            if i > 0 and i < len(route) - 1:
                cost_with += cost_matrix[route[i-1][0] + 1][route[i+1][0] + 1]
            
            customer_route_impact[cust_idx] += (cost_without - cost_with)
    
    # Normalize impact by vehicle count
    vehicle_count = np.maximum(vehicle_count, 1)
    customer_route_impact /= vehicle_count
    
    # Combine scores: weighted sum of risk exposure, mismatch, and route impact
    weights = np.array([0.4, 0.3, 0.3])
    scores = np.zeros(N)
    scores += weights[0] * inventory_risk
    scores += weights[1] * supply_demand_mismatch
    scores += weights[2] * customer_route_impact
    
    # Create list of (customer_index, score) tuples
    score_list = [(i, float(scores[i])) for i in range(N)]
    score_list.sort(key=lambda x: x[1], reverse=True)
    
    # Determine number of removals based on problem size and randomness
    k = max(1, min(N // 4, int(rng.integers(1, N // 3 + 1))))
    
    return score_list, k
"""

    inventory_code = """
def destroy_scoring(solution, data, rng):
    shipments = solution['shipments']
    routes = solution['routes']
    T, V, N = shipments.shape
    
    # Initialize scores
    scores = [[[0.0 for _ in range(N)] for _ in range(V)] for _ in range(T)]
    
    # Calculate inventory levels
    cust_inv = np.zeros((N, T))
    depot_inv = np.zeros(T)
    cust_inv[:, 0] = data['cust_I0']
    depot_inv[0] = data['depot_I0']
    
    # Calculate inventory levels for each period
    for t in range(T):
        if t > 0:
            cust_inv[:, t] = cust_inv[:, t-1] - data['cust_demand'][:, t-1]
            depot_inv[t] = (
                depot_inv[t-1]
                + data['depot_prod'][t-1]
                - np.sum(data['cust_demand'][:, t-1])
            )
        
        # Add shipments to inventory
        for v in range(V):
            for i in range(N):
                cust_inv[i, t] += shipments[t, v, i]
                depot_inv[t] -= shipments[t, v, i]
    
    # Calculate penalties for inventory violations
    cust_violation = np.zeros((N, T))
    depot_violation = np.zeros(T)
    
    for t in range(T):
        for i in range(N):
            if cust_inv[i, t] < 0:
                # Penalty for shortage
                cust_violation[i, t] = (
                    -cust_inv[i, t] * data['cust_inv_cost'][i] * 15
                )
            elif cust_inv[i, t] > data['cust_Lmax'][i]:
                # Penalty for excess
                cust_violation[i, t] = (
                    (cust_inv[i, t] - data['cust_Lmax'][i])
                    * data['cust_inv_cost'][i] * 5
                )
        
        if depot_inv[t] < 0:
            # Penalty for depot shortage
            depot_violation[t] = (
                -depot_inv[t] * data['depot_inv_cost'] * 15
            )
        elif depot_inv[t] > data['depot_I0'] * 10:
            # Penalty for depot excess (assumed max = 10x initial)
            depot_violation[t] = (
                (depot_inv[t] - data['depot_I0'] * 10)
                * data['depot_inv_cost'] * 5
            )
    
    # Calculate travel costs for each route
    travel_costs = np.zeros((T, V))
    for t in range(T):
        for v in range(V):
            if t in routes and v in routes[t]:
                route = routes[t][v]
                total_cost = 0
                prev_node = 0  # depot
                
                for customer, quantity in route:
                    total_cost += data['cost_matrix'][prev_node][customer + 1]
                    prev_node = customer + 1
                
                total_cost += data['cost_matrix'][prev_node][0]  # return to depot
                travel_costs[t, v] = total_cost
    
    # Calculate scores for each shipment element
    for t in range(T):
        for v in range(V):
            for i in range(N):
                if shipments[t, v, i] > 0:
                    # Base score components
                    violation_cost = cust_violation[i, t] + depot_violation[t]
                    travel_cost = travel_costs[t, v]
                    
                    # Higher score = higher potential benefit from removal
                    score = violation_cost + travel_cost
                    
                    # Random diversification factor
                    random_factor = rng.uniform(0.5, 1.5)
                    score *= random_factor
                    
                    # Weight shipment quantity
                    score += shipments[t, v, i] * 0.05
                    score += shipments[t, v, i] * 0.08
                    
                    scores[t][v][i] = score
    
    # Determine number of elements to remove (1 to 3% of non-zero elements)
    non_zero_count = np.count_nonzero(shipments)
    k = max(1, min(3, int(non_zero_count * 0.03)))
    
    return scores, k
"""

    print("===== LNS HEURISTIC EVALUATION TESTS =====")

    dotenv.load_dotenv()

    eoh_experiment_file = f"U_C_N?_Q18_T6"

    #os.environ["CURRENT_EXPERIMENT"] = (
    #        r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling"
    #        r"\eoh_results\results_test_lns_multi_robot_upmp"
    #    )

    # --------------------------------------------------
    # Shared base parameters
    # --------------------------------------------------
    base_paras = dict(
            eoh_experiment_file=eoh_experiment_file,
            lns_paras={
                "num_iters": 1000,
                "runtime_limit": 10,
                "no_improvement_time": 10,
                "no_improvement_iters": 1000,
            },
            lns_paras_second={
                "num_iters": 1000,
                "runtime_limit": 10,
                "no_improvement_time": 10,
                "no_improvement_iters": 1000,

            },
            alns_cycles_iters=100000,
            alns_cycles_timeout=60,
    )

    # ==================================================
    # 2) INDIVIDUAL MODE — VRP
    # ==================================================
    print(f"\n--- TEST: Individual {VRP_PROBLEM} heuristic ---")

    paras_vrp = Paras()
    paras_vrp.set_paras(
            **base_paras,
            target_subproblem=VRP_PROBLEM,
            coupled_problems=[
                INVENTORY_PROBLEM,
                VRP_PROBLEM,
            ],
    )

    #lns_vrp = LNS_IRP_CONSTANT(paras=paras_vrp)
    #fitness_vrp, _ = lns_vrp.evaluate(vrp_code)

    #print("VRP-only fitness:", fitness_vrp)

    # ==================================================
    # 1) INDIVIDUAL MODE
    # ==================================================
    #print(f"\n--- TEST: Individual {INVENTORY_PROBLEM} heuristic ---")

    paras_move = Paras()
    paras_move.set_paras(
        **base_paras,
        target_subproblem=INVENTORY_PROBLEM,
        coupled_problems=[
            INVENTORY_PROBLEM,
            VRP_PROBLEM,
        ],
    )

    #lns_move = LNS_IRP_CONSTANT(paras=paras_move)
    #fitness_move, _ = lns_move.evaluate(inventory_code)

    #print("MOVE-only fitness:", fitness_move)

    # ==================================================
    # 3) COUPLED MODE —  heuristics together
    # ==================================================
    print(f"\n--- TEST 3: Coupled {INVENTORY_PROBLEM} + {VRP_PROBLEM} heuristics ---")

    paras_coupled = Paras()
    paras_coupled.set_paras(
            **base_paras,
            target_subproblem=COUPLED_PROBLEM,
    )

    coupled_code_strings = {
            INVENTORY_PROBLEM: inventory_code,
            VRP_PROBLEM: vrp_code,
    }

    lns_coupled = LNS_IRP_CONSTANT(paras=paras_coupled)

    # This assumes you added evaluate_coupled(...)
    fitness_coupled, _ = lns_coupled.evaluate(coupled_code_strings)
    print("COUPLED fitness:", fitness_coupled)
    print("\n===== ALL TESTS COMPLETED =====")

