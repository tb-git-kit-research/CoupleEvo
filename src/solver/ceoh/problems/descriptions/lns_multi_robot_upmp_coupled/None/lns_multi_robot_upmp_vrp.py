# ----------------------------- Standard Library -----------------------------
import copy
import math
import os
import random
import types
import numpy as np

# ----------------------------- Third-Party Libraries -----------------------------
import networkx as nx
import numpy.random as rnd
rnd.default_rng(seed=42)

from dotenv import load_dotenv

from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import NoImprovement

# ----------------------------- Project / Local Modules -----------------------------
from libraries.lns.base_class import LNS_Base
from libraries.lns.core.StopCriteria import (
    MaxIterationsAndMaxRuntime,
    NoImprovementTime,
    OrStop,
)

from problems.multibay_reshuffeling.bay.warehouse import Warehouse

from solver.ceoh.utils.getParas import Paras

from util.mr_to_ceoh_util import (
    create_virtual_lane,
    get_access_directions,
    load_experiments,
)

from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar_loaded_move_time.run import HANDLING_TIME

from problems.multibay_reshuffeling.mr_util.search_move_sequence import (
    compute_id_dependencies,
)

from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar_loaded_move_time.run import (
    astar_multibay_premarshalling,
)
from libraries.lns.wrapper.lns_multi_robot_upmp_move_search import (
    HEURISTIC_CODE_INITIAL,
)

# ----------------------------- Problem State -----------------------------
class ProblemState:
    def __init__(self, solution, data, parent_problem_state=None, unassigned_moves=None, astar_solution=None):
        self.solution = solution
        self.data = data
        self.parent_solution = parent_problem_state.solution if parent_problem_state else None
        self.tabu_dict = parent_problem_state.tabu_dict if parent_problem_state else {}

        # Unassigned moves list
        self.unassigned = unassigned_moves if unassigned_moves else []

        # Initialize dependencies only at root level
        if parent_problem_state is None:
            self.data.move_dependencies = compute_id_dependencies(solution, data, astar_solution)

    def objective(self):
        """Main optimization objective: makespan."""
        return self.get_total_makespan() # self.get_total_travel_time() #

    def get_total_makespan(self):
        """Maximum end_time across all moves."""
        return max(route[-1]["end_time"] for route in self.solution if route)

    def get_total_travel_time(self):
        """Loaded + unloaded move time + handling overhead."""
        return (
            self.get_loaded_move_time()
            + self.get_unloaded_move_time()
            + HANDLING_TIME * 2 * self.get_move_number()
        )

    def get_loaded_move_time(self):
        """Travel time needed for loaded movements."""
        return sum(
            self.data.ap_distance[m["move"][0]][m["move"][1]]
            for route in self.solution for m in route
        )

    def get_unloaded_move_time(self):
        """Travel time needed for unloaded movements."""
        total = 0
        for route in self.solution:
            for i in range(len(route) - 1):
                total += self.data.ap_distance[
                    route[i]["move"][1]
                ][
                    route[i + 1]["move"][0]
                ]
        return total

    def get_move_number(self):
        """Return the number of valid moves (ignore moves where source == destination)."""
        return sum(
            1
            for route in self.solution
            for m in route
            if m["move"][0] != m["move"][1]
        )

# ----------------------------- Initial State -----------------------------
def initial_state(data, paras, instance_seed):
    heuristic_module = types.ModuleType("heuristic_module")
    exec(HEURISTIC_CODE_INITIAL, heuristic_module.__dict__)
    result = astar_multibay_premarshalling(heuristic_module, data)
    if result["solution"] is None:
        raise Exception("No solution found in A*.")

    number_robots = len(data.robot_aps)
    robot_aps = data.robot_aps
    move_id = 1
    initial_routing = []

    for r in range(number_robots):
        route = []
        route.append({
            "id": 0,
            "move": [robot_aps[r],robot_aps[r]],
            "start_time": 0,  # placeholder
            "end_time": 0,  # placeholder
            "robot": r,
        })
        if r == 0:  # assign A* solution to robot 0
            for move in result["solution"]:
                route.append({
                    "id": move_id,
                    "move": list(move),
                    "start_time": 0,  # placeholder
                    "end_time": 0,    # placeholder
                    "robot": r,
                })
                move_id += 1
        initial_routing.append(route)

    # --- Step 1: baseline timing on each route ---
    for route in initial_routing:
        if not route:
            continue

        #dummy move
        route[0]["start_time"] = 0
        route[0]["end_time"] = 0

        # subsequent moves
        for i in range(1, len(route)):
            prev, curr = route[i - 1], route[i]
            travel  = data.ap_distance[prev["move"][1]][curr["move"][0]]      # unloaded move time
            service = data.ap_distance[curr["move"][0]][curr["move"][1]]      # loaded move time
            curr["start_time"] = prev["end_time"] + travel
            curr["end_time"]   = curr["start_time"] + service + 2 * HANDLING_TIME

    return ProblemState(initial_routing, data)

# ----------------------------- Destroy (Routing) -----------------------------
def destroy(current: ProblemState, rng: rnd.Generator) -> ProblemState:
    destroyed = copy.deepcopy(current)

    scores, k = destroy_scoring(destroyed.solution, destroyed.data, rng)

    destroyed = apply_scoring(destroyed, scores, k)

    return destroyed


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

def apply_scoring(state: ProblemState, scores, k: int) -> ProblemState:
    """
    Removes the top-k scored moves by ID from `state.solution`
    and adds them to `state.unassigned`.

    `scores` is a list of (move_id, score).
    """

    if not scores or k <= 0:
        return state

    sorted_scores = sorted(scores, key=lambda t: t[1], reverse=True)

    picked_ids = []
    for move_id, _ in sorted_scores:
        picked_ids.append(move_id)
        if len(picked_ids) == k:
            break

    removed_moves = []

    for move_id in picked_ids:
        for route in state.solution:
            for move in route:
                if move["id"] == move_id:
                    route.remove(move)
                    removed_moves.append(move)
                    break
            else:
                continue
            break

    state.unassigned.extend(removed_moves)

    return state

# ----------------------------- Repair (Routing) -----------------------------
def repair(destroyed: ProblemState, rng: rnd.Generator) -> ProblemState:
    destroyed_solution = copy.deepcopy(destroyed.solution)
    unassigned = copy.deepcopy(destroyed.unassigned)

    def backtrack(current_solution, remaining_moves):
        if not remaining_moves:
            return current_solution

        move = remaining_moves[0]
        rest = remaining_moves[1:]

        options = get_insert_options(destroyed, current_solution, move)
        scores = insert_cost(rng, options, destroyed)
        sorted_options = [opt for _, opt in sorted(zip(scores, options), key=lambda x: x[0])]

        for option in sorted_options:
            result = backtrack(option, rest)
            if result:
                return result
        return None

    repaired = backtrack(destroyed_solution, unassigned)
    if not repaired:
        raise Exception("Repair failed: no valid sequence.")
    repaired_state = ProblemState(repaired, destroyed.data, parent_problem_state=destroyed)

    return repaired_state

def get_insert_options(destroyed, current_solution, move):
    valid = []

    for r in range(len(current_solution)):
        for pos in range(1, len(current_solution[r]) + 1):
            temp_solution = copy.deepcopy(current_solution)

            temp_solution[r] = temp_solution[r][:pos] + [copy.deepcopy(move)] + temp_solution[r][pos:]

            assign_robot_ids(temp_solution)

            if check_constraints(temp_solution, destroyed.data):
                valid.append(temp_solution)

    return valid

def assign_robot_ids(solution):
    """
    Update the 'robot' field of all moves based on their route index.
    solution: list of routes, each route = list of move dicts.
    """
    for r, route in enumerate(solution):
        for m in route:
            m["robot"] = r
    return solution

def check_constraints(solution, data):
    # Flatten moves (id -> move dict)
    moves = {m['id']: m for route in solution for m in route}

    id_to_move = {m["id"]: m for route in solution for m in route}

    # --- Step 1: baseline timing on each route ---
    for route in solution:
        if not route:
            continue

        # dummy move
        route[0]["start_time"] = 0
        route[0]["end_time"] = 0

        # subsequent moves
        for i in range(1, len(route)):
            prev, curr = route[i - 1], route[i]
            travel = data.ap_distance[prev["move"][1]][curr["move"][0]]  # unloaded move time
            service = data.ap_distance[curr["move"][0]][curr["move"][1]]  # loaded move time
            curr["start_time"] = prev["end_time"] + travel
            curr["end_time"] = curr["start_time"] + service + 2 * HANDLING_TIME

    # --- Step 2: precedence constraints ---
    all_pairs = []
    for prio in data.move_dependencies.values():
        for sublist in prio.values():
            all_pairs.extend(sublist)

    # Check for cycles
    if has_structural_violation(solution, all_pairs):
        #print("Structural precedence violation (cycle).")
        return False

    changed = True
    max_iter, iters = 10000, 0
    while changed:
        changed = False
        iters += 1

        # --- start-start constraints unequal prio ---
        for i_id, j_id in data.move_dependencies.get('unequal_prio', {}).get('start-start', []):
            if i_id not in moves or j_id not in moves:
                continue
            mi, mj = id_to_move[i_id], id_to_move[j_id]

            if mi["robot"] == mj["robot"] and not mi["start_time"] + HANDLING_TIME <= mj["start_time"] :
                return False
            if not mi["start_time"] + HANDLING_TIME <= mj["start_time"] :
                delay = mi["start_time"] + HANDLING_TIME - mj["start_time"]
                solution = shift_route(solution, mj, delay, data)
                moves = {m['id']: m for route in solution for m in route}
                changed = True

        # --- end-end constraints unequal prio ---
        for i_id, j_id in data.move_dependencies.get('unequal_prio', {}).get('end-end', []):
            if i_id not in moves or j_id not in moves:
                continue
            mi, mj = id_to_move[i_id], id_to_move[j_id]
            if mi["robot"] == mj["robot"] and not mi["end_time"] + HANDLING_TIME <= mj["end_time"]:
                return False
            if not mi["end_time"] + HANDLING_TIME <= mj["end_time"] :
                delay = mi["end_time"] + HANDLING_TIME - mj["end_time"]
                solution = shift_route(solution, mj, delay, data)
                moves = {m['id']: m for route in solution for m in route}
                changed = True

        # --- start-end constraints unequal and equal prio ---
        for i_id, j_id in (data.move_dependencies.get('unequal_prio', {}).get('start-end', []) +
                           data.move_dependencies.get('equal_prio', {}).get('start-end', [])):
            if i_id not in moves or j_id not in moves:
                continue
            mi, mj = id_to_move[i_id], id_to_move[j_id]
            if mi["robot"] == mj["robot"] and not mi["start_time"] + 2 * HANDLING_TIME <= mj["end_time"]:
                return False
            if not mi["start_time"] + 2 * HANDLING_TIME <= mj["end_time"]:
                delay = mi["start_time"] + 2 * HANDLING_TIME - mj["end_time"]
                solution = shift_route(solution, mj, delay, data)
                moves = {m['id']: m for route in solution for m in route}
                changed = True

        # --- end-start constraints unequal and equal prio ---
        for i_id, j_id in (data.move_dependencies.get('unequal_prio', {}).get('end-start', [])
                           + data.move_dependencies.get('equal_prio', {}).get('end-start', [])):
            if i_id not in moves or j_id not in moves:
                continue
            mi, mj = id_to_move[i_id], id_to_move[j_id]
            if mi["robot"] == mj["robot"] and not mi["end_time"] <= mj["start_time"]:
                return False
            if not mi["end_time"] <= mj["start_time"]:
                delay = mi["end_time"] - mj["start_time"]
                solution = shift_route(solution, mj, delay, data)
                moves = {m['id']: m for route in solution for m in route}
                changed = True

        # --- start-start constraints equal prio ---
        for i_id, j_id in data.move_dependencies.get('equal_prio', {}).get('start-start', []):
            if i_id not in moves or j_id not in moves:
                continue
            mi, mj = id_to_move[i_id], id_to_move[j_id]
            if (mi["robot"] == mj["robot"]
                    and not (mi["start_time"] + HANDLING_TIME <= mj["start_time"] or
                             mj["start_time"] + HANDLING_TIME <= mi["start_time"])) :
                return False
            if not (mi["start_time"] + HANDLING_TIME <= mj["start_time"] or
                    mj["start_time"] + HANDLING_TIME <= mi["start_time"]):

                delay_mi = mj["start_time"] + HANDLING_TIME - mi["start_time"]
                delay_mj = mi["start_time"] + HANDLING_TIME - mj["start_time"]

                if delay_mj <= delay_mi:
                    solution = shift_route(solution, mj, delay_mj, data)
                else:
                    solution = shift_route(solution, mi, delay_mi, data)

                moves = {m['id']: m for route in solution for m in route}
                changed = True

        # --- end-end constraints equal prio ---
        for i_id, j_id in data.move_dependencies.get('equal_prio', {}).get('end-end', []):
            if i_id not in moves or j_id not in moves:
                continue
            mi, mj = id_to_move[i_id], id_to_move[j_id]
            if (mi["robot"] == mj["robot"]
                and not (mi["end_time"] + HANDLING_TIME <= mj["end_time"] or
                         mj["end_time"] + HANDLING_TIME <= mi["end_time"])):
                return False
            if not (mi["end_time"] + HANDLING_TIME <= mj["end_time"] or
                    mj["end_time"] + HANDLING_TIME <= mi["end_time"]):

                delay_mi = mj["end_time"] + HANDLING_TIME - mi["end_time"]
                delay_mj = mi["end_time"] + HANDLING_TIME - mj["end_time"]

                if delay_mj <= delay_mi:
                    solution = shift_route(solution, mj, delay_mj, data)
                else:
                    solution = shift_route(solution, mi, delay_mi, data)

                moves = {m['id']: m for route in solution for m in route}
                changed = True

    if iters == max_iter:
        print("Warning: precedence propagation did not converge. Possibly infeasible.")
        return False
    return True

def has_structural_violation(solution, precedences):
    G = nx.DiGraph()

    # Add precedence edges
    for i, j in precedences:
        G.add_edge(i, j)

    # Add within-route order edges
    for route in solution:
        for a, b in zip(route, route[1:]):
            G.add_edge(a["id"], b["id"])

    # If graph is not a DAG, then infeasible
    return not nx.is_directed_acyclic_graph(G)

def shift_route(solution, successor_move, delay, data):
    """
    Shift successor_move and all later moves in its route by 'delay'.
    Recompute start_time and end_time correctly using travel + service + handling.
    """
    for route in solution:
        for idx, m in enumerate(route):
            if m["id"] == successor_move["id"]:
                # Apply delay only from this move onward
                route[idx]["start_time"] += delay
                service = data.ap_distance[route[idx]["move"][0]][route[idx]["move"][1]]
                route[idx]["end_time"] = route[idx]["start_time"] + service + 2 * HANDLING_TIME

                # Recompute the rest of the route sequentially
                for k in range(idx+1, len(route)):
                    prev, curr = route[k - 1], route[k]
                    travel  = data.ap_distance[prev["move"][1]][curr["move"][0]]   # unloaded
                    service = data.ap_distance[curr["move"][0]][curr["move"][1]]   # loaded
                    curr["start_time"] = prev["end_time"] + travel
                    curr["end_time"]   = curr["start_time"] + service + 2 * HANDLING_TIME
    return solution

def insert_cost(rng, solution_options, destroyed):
    """
    Compute cost for each solution option based on objective.
    """
    scores = []
    for s in solution_options:
        # Wrap in ProblemState to compute objective
        temp_state = ProblemState(
            solution=s,
            data=destroyed.data,
            parent_problem_state=destroyed
        )
        cost = temp_state.objective()

        scores.append(cost)

    return scores


# ----------------------------- ALNS Setup -----------------------------

def load_data(instance_seed=0, paras=None):
    eoh_experiment_file = paras.eoh_experiment_file
    configs, _ = load_experiments(eoh_experiment_file)
    config = configs[instance_seed]
    access_directions = get_access_directions(config)
    path = os.path.join(os.getenv("BASE_PATH"), "data", "mr_examples")
    wh = Warehouse(os.path.join(path, config["layout_file"]), access_directions)
    wh.virtual_lanes = create_virtual_lane(config)

    wh.robot_number = int(paras.eoh_experiment_file.split("robots_")[1].split("_")[0].strip(".json")) #robots_2_a
    wh.robot_aps = set_random_robots(robot_number=wh.robot_number, seed=instance_seed,virtual_lanes=wh.virtual_lanes)
    return wh

def set_random_robots(robot_number, seed, virtual_lanes):
    random.seed(seed)

    nr_of_aps = len(virtual_lanes)

    robot_aps = []
    for index, _ in enumerate(range(robot_number)):
        choice = int(random.choice(range(nr_of_aps)))

        # Don't select double numbers
        while choice in robot_aps:
            #print('Double robot')
            choice = int(random.choice(range(nr_of_aps)))

        robot_aps.append(choice)
    return robot_aps

def stop(num_iters: int, runtime_limit: int, no_improvement_time: int, no_improvement_iters: int):
    return OrStop(
        NoImprovement(no_improvement_iters),
        NoImprovementTime(no_improvement_time),
        MaxIterationsAndMaxRuntime(num_iters, runtime_limit),
    )

def select(num_destroy: int, num_repair: int):
    return RouletteWheel([5, 2, 1, 0.5], 0.8, num_destroy, num_repair)

def accept(init: ProblemState, num_iters: int):
    return RecordToRecordTravel.autofit(
        init.objective(), 0.02, 0, num_iters
    )

def info(_):
    print("Routing-Phase: Kein spezieller Info-Output.")


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    load_dotenv()

    print("Starting ALNS Route Optimization...")
    lns = LNS_Base(problem="lns_multi_robot_upmp_vrp")

    paras_instance = Paras()
    paras_instance.set_paras(
        eoh_experiment_file="exp_bay_5_wh_2_fill_0.8_seed_0-9_priority_5_robots_2_ad_1010.json",
        lns_paras={
            "num_iters": 1000,
            "runtime_limit": 30,
            "no_improvement_time": 10,
            "no_improvement_iters": 3000,
        },
    )

    lns.init_lns(1, paras=paras_instance)

    lns.run()
    result = lns.get_result()
    lns.plot_objective()
    id_lists = [[item["id"] for item in sublist] for sublist in result.solution]

    print(id_lists)
    print( [item["end_time"] for sublist in result.solution for item in sublist])

    print(result.solution)
    print(result.objective())


    for r in result.solution:
        for m in r:
            print(m["id"], m["start_time"], m["end_time"])
