import inspect
import time
import types
from multiprocessing import Process, Queue

import dotenv

from libraries.lns.wrapper.run_alns_cycle_move_route import run_alternating_alns
import libraries.lns.wrapper.lns_multi_robot_upmp_move_search
from solver.ceoh.utils.getParas import Paras
from solver.ceoh.problems.descriptions.lns_multi_robot_upmp_move_search.prompts import GetPrompts

from libraries.lns.base_class import LNS_Base

####################################################
# Config
MOVE_PROBLEM = 'lns_multi_robot_upmp_move_search'
VRP_PROBLEM = 'lns_multi_robot_upmp_vrp'
TIMEOUT_SECONDS = 120
N_WORKERS = 10

HUMAN_CODE_STRING = """
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
####################################################

def _build_heuristic_module(code_string: str) -> types.ModuleType:
    """Baut das Heuristik-Modul wie in deinem evaluate() zusammen."""
    imports = (
        "import copy\n"
        "import numpy as np\n"
        "import math\n"
        f"from libraries.lns.wrapper.{MOVE_PROBLEM} import *\n\n"
    )
    destroy_wrapper = (
        "\n\n"
        "def destroy(current: ProblemState, rng: rnd.Generator) -> ProblemState:\n"
        "\tdestroyed = ProblemState(copy.deepcopy(current.solution), current.data, current)\n"
        "\tscores, k = destroy_scoring(destroyed.solution, destroyed.data, rng)\n"
        "\tdestroyed_solution = apply_scoring(destroyed, scores, k)\n"
        "\treturn ProblemState(solution=destroyed_solution, data=current.data, parent_problem_state=current) \n\n"
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
            heuristic_module = _build_heuristic_module(code_string)

            lns_move = LNS_Base(MOVE_PROBLEM)
            lns_route = LNS_Base(VRP_PROBLEM)

            lns_move.set_destroy(heuristic_module.destroy)

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

        self.lns_move = LNS_Base(MOVE_PROBLEM)
        self.lns_route = LNS_Base(VRP_PROBLEM)

        self.prompts = GetPrompts()

        self.paras = paras

        #self.ref_solutions = None,

        self.seeds = list(range(0, 10))

        self.human_code_string = inspect.getsource(libraries.lns.wrapper.lns_multi_robot_upmp_move_search.destroy_scoring)

        global N_WORKERS
        N_WORKERS = min(len(self.seeds), 10)


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
    code_string = """  
def destroy_scoring(current_solution, data, rng):
    scores = []
    total_moves = len(current_solution)
    
    access_point_utilization = {ap: 0 for ap in set(ap for move in current_solution for ap in move)}
    for move in current_solution:
        pick_up, drop_off = move
        access_point_utilization[pick_up] += 1
        access_point_utilization[drop_off] += 1
    
    for i, move in enumerate(current_solution):
        pick_up, drop_off = move
        distance_score = data.ap_distance[pick_up][drop_off]
        
        dependency_complexity = sum(
            sum(1 for dep in dependencies if i in dep)
            for dep_type in ['equal_prio', 'unequal_prio']
            for key in ['end-end', 'start-end', 'end-start', 'start-start']
            for dependencies in [data.move_dependencies[dep_type][key]]
        )
        
        utilization_penalty = access_point_utilization[pick_up] + access_point_utilization[drop_off]
        
        random_score = rng.random()
        
        total_pickup_score = (
            distance_score
            + dependency_complexity * 2
            - utilization_penalty * 0.5
            - random_score
        )
        total_dropoff_score = (
            distance_score
            + dependency_complexity * 2
            - utilization_penalty * 0.5
            - random_score
        )
        
        scores.append([total_pickup_score, total_dropoff_score])
    
    k = max(1, int(total_moves * 0.1))  # Remove 10% of moves but at least one
    
    return scores, k
"""

    #subprocess.run(["pip", "install", "."], stdout=subprocess.DEVNULL)
    dotenv.load_dotenv()

    eoh_experiment_file = "exp_bay_5_wh_2_fill_0.8_seed_0-9_priority_5_robots_2_ad_1010.json"

    paras = Paras()
    # add eoh_experiment_file to paras
    paras.set_paras(
        eoh_experiment_file=eoh_experiment_file,
        lns_paras = {
            "num_iters" : 10,
            "runtime_limit" : 10,
            "no_improvement_time" : 10,
            "no_improvement_iters" : 10,
        },
        lns_paras_second = {
            "num_iters": 90,
            "runtime_limit": 10,
            "no_improvement_time": 10,
            "no_improvement_iters": 90,
        },
        alns_cycles_iters = 10,
        alns_cycles_timeout = 60
    )

    lns = LNS_MOVE_ROUTE_CONSTANT(code_string, paras = paras)
    lns.evaluate(code_string)
    print("Done")
