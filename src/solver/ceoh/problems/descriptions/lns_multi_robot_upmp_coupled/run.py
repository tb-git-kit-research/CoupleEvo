import inspect
import os
import time
import types
from multiprocessing import Process, Queue

import dotenv

from libraries.lns.wrapper.run_alns_cycle_move_route import run_alternating_alns

from solver.ceoh.problems.descriptions.lns_multi_robot_upmp_vrp import run as run_vrp
from solver.ceoh.problems.descriptions.lns_multi_robot_upmp_move_search import run as run_move

from solver.ceoh.problems.descriptions.lns_multi_robot_upmp_coupled.prompts import GetPrompts

from solver.ceoh.utils.get_coupled_heuristics import get_code_strings

from solver.ceoh.utils.getParas import Paras

from libraries.lns.base_class import LNS_Base

import libraries.lns.wrapper.lns_multi_robot_upmp_move_search
import libraries.lns.wrapper.lns_multi_robot_upmp_vrp

####################################################
# Config
MOVE_PROBLEM = 'lns_multi_robot_upmp_move_search'
VRP_PROBLEM = 'lns_multi_robot_upmp_vrp'
COUPLED_PROBLEM = 'lns_multi_robot_upmp_coupled'
SUBPROBLEMS = [MOVE_PROBLEM, VRP_PROBLEM]
TIMEOUT_SECONDS = 85
N_WORKERS = 10
####################################################

def solve_seed_worker(task_queue, result_queue, code_string, paras):
    while True:
        seed = task_queue.get()
        if seed is None:
            break
        try:
            print(f"Evaluate heuritsic generated for {paras.target_subproblem}")
            print(code_string)

            # liste von lns wrappern
            lns_move = LNS_Base(MOVE_PROBLEM)
            lns_route = LNS_Base(VRP_PROBLEM)

            print(f"Heuritsic generated for {paras.target_subproblem}")

            move_code = None
            vrp_code = None

            if paras.target_subproblem == COUPLED_PROBLEM:
                # both heuristics passed explicitly
                if code_string is None:
                    raise ValueError("Coupled problem requires coupled_code_strings")

                move_code = code_string.get(MOVE_PROBLEM)
                vrp_code = code_string.get(VRP_PROBLEM)

            elif paras.target_subproblem == MOVE_PROBLEM:
                stored_code_strings = get_code_strings(paras)
                move_code = code_string
                vrp_code = stored_code_strings.get(VRP_PROBLEM)

            elif paras.target_subproblem == VRP_PROBLEM:
                stored_code_strings = get_code_strings(paras)
                vrp_code = code_string
                move_code = stored_code_strings.get(MOVE_PROBLEM)

            if move_code is not None:
                heuristic = run_move._build_heuristic_module(move_code)
                lns_move.set_destroy(heuristic.destroy)
                print("Registered MOVE destroy heuristic")

            if vrp_code is not None:
                heuristic = run_vrp._build_heuristic_module(vrp_code)
                lns_route.set_destroy(heuristic.destroy)
                print("Registered VRP destroy heuristic")

            paras_move = Paras()
            paras_move.set_paras(
                eoh_experiment_file=paras.eoh_experiment_file,
                lns_paras=paras.lns_paras
            )

            paras_route = Paras()
            paras_route.set_paras(
                eoh_experiment_file=paras.eoh_experiment_file,
                lns_paras=paras.lns_paras_second
            )

            results_dict = run_alternating_alns(cycles=paras.alns_cycles_iters,
                                                paras_move=paras_move,
                                                paras_route=paras_route,
                                                instance_seed=seed,
                                                lns_move=lns_move,
                                                lns_route=lns_route,
                                                timeout = paras.alns_cycles_timeout)

            fitness = min(results_dict["objective_history"])

            result_queue.put({"seed": seed, "fitness": fitness, "results": results_dict})
        except Exception as e:
            result_queue.put({"seed": seed, "fitness": None, "error": str(e)})

class LNS_MOVE_ROUTE_CONSTANT:
    def __init__(self, code_string=None, paras=None):

        self.paras = paras

        #self.ref_solutions = None,

        self.seeds = list(range(0, 10))

        self.prompts = GetPrompts()

        self.human_code_string = [inspect.getsource(libraries.lns.wrapper.lns_multi_robot_upmp_move_search.destroy_scoring),
                                  inspect.getsource(libraries.lns.wrapper.lns_multi_robot_upmp_vrp.destroy_scoring)]

        global N_WORKERS
        N_WORKERS = min(len(self.seeds), 10)

        if paras.n_workers:
            N_WORKERS = paras.n_workers

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

        fitness_values = [r["fitness"] for r in results if r and r.get("fitness") is not None]
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else None
        print("Durchschnitt Fitness:", round(avg_fitness, 4), "\n")


        return avg_fitness, results

if __name__ == "__main__":

    print("Start LNS for CYCLE MOVE + ROUTE")
    vrp_code = """  

def destroy_scoring(current_solution, data, rng):
    scores = []

    for route in current_solution:
        for pos, m in enumerate(route):
            if m["id"] == 0:
                continue

            move_id = m["id"]
            random_score = rng.random()
            scores.append((move_id, random_score))

    num_nodes = sum(len(route) for route in current_solution)
    frac = rng.uniform(0.1, 0.4)
    k = math.ceil(frac * num_nodes)

    k = max(1, k)
    return scores, k
"""

    move_code = """
def destroy_scoring(current_solution, data, rng):
    scores = []
    ap_frequency = {ap: 0 for move in current_solution for ap in move}

    for i, move in enumerate(current_solution):
        pickup = move[0]
        dropoff = move[1]

        pu_freq = ap_frequency[pickup]
        do_freq = ap_frequency[dropoff]

        seq_bonus = rng.random() * i

        sp = (rng.random() * 0.05
              + seq_bonus
              + pu_freq * 0.02
              )

        sd = (rng.random() * 0.05
              + data.ap_distance[pickup, dropoff] * 0.1
              + seq_bonus
              + do_freq * 0.01
              )

        scores.append([sp, sd])

    k = int(np.ceil(0.125 * len(current_solution)))

    return scores, k
"""


    print("===== LNS HEURISTIC EVALUATION TESTS =====")

    dotenv.load_dotenv()

    eoh_experiment_file = (
            "exp_bay_5_wh_2_fill_0.8_seed_0-9_priority_5_robots_2_ad_1010.json"
        )

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
                "num_iters": 10,
                "runtime_limit": 10,
                "no_improvement_time": 10,
                "no_improvement_iters": 10,
            },
            lns_paras_second={
                "num_iters": 90,
                "runtime_limit": 10,
                "no_improvement_time": 10,
                "no_improvement_iters": 90,
            },
            alns_cycles_iters=100,
            alns_cycles_timeout=60,
    )

    # ==================================================
    # 1) INDIVIDUAL MODE — MOVE heuristic only
    # ==================================================
    print("\n--- TEST 1: Individual MOVE heuristic ---")

    paras_move = Paras()
    paras_move.set_paras(
        **base_paras,
        target_subproblem=MOVE_PROBLEM,
        coupled_problems=[
            MOVE_PROBLEM,
            VRP_PROBLEM,
        ],
    )

    #lns_move = LNS_MOVE_ROUTE_CONSTANT(paras=paras_move)
    #fitness_move, _ = lns_move.evaluate(move_code)

    #print("MOVE-only fitness:", fitness_move)

    # ==================================================
    # 2) INDIVIDUAL MODE — VRP heuristic only
    # ==================================================
    print("\n--- TEST 2: Individual VRP heuristic ---")

    paras_vrp = Paras()
    paras_vrp.set_paras(
            **base_paras,
            target_subproblem=VRP_PROBLEM,
            coupled_problems=[
                MOVE_PROBLEM,
                VRP_PROBLEM,
            ],
    )

    #lns_vrp = LNS_MOVE_ROUTE_CONSTANT(paras=paras_vrp)
    #fitness_vrp, _ = lns_vrp.evaluate(vrp_code)

    #print("VRP-only fitness:", fitness_vrp)

    # ==================================================
    # 3) COUPLED MODE — MOVE + VRP heuristics together
    # ==================================================
    print("\n--- TEST 3: Coupled MOVE + VRP heuristics ---")

    paras_coupled = Paras()
    paras_coupled.set_paras(
            **base_paras,
            target_subproblem=COUPLED_PROBLEM,
    )

    coupled_code_strings = {
            MOVE_PROBLEM: move_code,
            VRP_PROBLEM: vrp_code,
    }

    lns_coupled = LNS_MOVE_ROUTE_CONSTANT(paras=paras_coupled)

    # This assumes you added evaluate_coupled(...)
    fitness_coupled, _ = lns_coupled.evaluate(coupled_code_strings)

    print("COUPLED fitness:", fitness_coupled)

    print("\n===== ALL TESTS COMPLETED =====")

