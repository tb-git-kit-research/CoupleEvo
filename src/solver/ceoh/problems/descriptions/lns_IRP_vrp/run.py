import inspect
import os
import time
import types
from multiprocessing import Process, Queue

import dotenv
import numpy as np

from problems.inventory_routing.get_HGS_objective import load_all_hgs_stats

from problems.inventory_routing.get_instance import load_full_skalnes_instances

from solver.ceoh.problems.descriptions.lns_IRP_coupled.run import _build_heuristic_module_vrp, \
    _build_heuristic_module_inventory

import libraries.lns.wrapper.lns_IRP_vrp
from libraries.lns.wrapper.run_alns_cycle_inventory_routing import run_alternating_alns

from solver.ceoh.problems.descriptions.lns_IRP_vrp.prompts import GetPrompts

from solver.ceoh.utils.get_coupled_heuristics import get_code_strings

from solver.ceoh.utils.getParas import Paras

from libraries.lns.base_class import LNS_Base

####################################################
# Config
INVENTORY_PROBLEM = 'lns_IRP_inventory'
VRP_PROBLEM = 'lns_IRP_vrp'
COUPLED_PROBLEM = 'lns_IRP_coupled'
SUBPROBLEMS = [INVENTORY_PROBLEM, VRP_PROBLEM]
TIMEOUT_SECONDS = 120
N_WORKERS = 10
####################################################


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

        self.human_code_string = inspect.getsource(libraries.lns.wrapper.lns_IRP_vrp.destroy_scoring)


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
"""

    print("===== LNS HEURISTIC EVALUATION TESTS =====")

    dotenv.load_dotenv()

    eoh_experiment_file = f"U_C_N25_Q?_T?"

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
                "num_iters": 100,
                "runtime_limit": 10,
                "no_improvement_time": 100,
                "no_improvement_iters": 100,
            },
            lns_paras_second={
                "num_iters": 100,
                "runtime_limit": 10,
                "no_improvement_time": 100,
                "no_improvement_iters": 100,
            },
            alns_cycles_iters=20,
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

    lns_vrp = LNS_IRP_CONSTANT(paras=paras_vrp)
    fitness_vrp, _ = lns_vrp.evaluate(vrp_code)

    print("VRP-only fitness:", fitness_vrp)


