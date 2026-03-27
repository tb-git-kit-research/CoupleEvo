import os
import types
import copy
import time
import heapq
import random

import numpy as np
import numpy.random as rnd
from alns.stop import NoImprovement

from dotenv import load_dotenv

from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from libraries.lns.base_class import LNS_Base
from libraries.lns.core.StopCriteria import MaxIterationsAndMaxRuntime, OrStop, NoImprovementTime

from problems.multibay_reshuffeling.bay.warehouse import Warehouse
from solver.ceoh.utils.getParas import Paras
from util.mr_to_ceoh_util import (
    load_experiments,
    get_access_directions,
    create_virtual_lane,
    convert_vl_to_list
)

from libraries.lns.wrapper.lns_upmp_helper import (
    get_heuristic_id,
    get_initial_solution_path,
    save_initial_solution,
    load_initial_solution
)

from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar_loaded_move_time.run import \
    astar_multibay_premarshalling
from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar_loaded_move_time.run import HANDLING_TIME

MAX_TABU_LENGTH = 0
TABU_PENALTY = 1_000_000

# ----------------------------- Heuristic -----------------------------
HEURISTIC_CODE_INITIAL = """
def score_state(state, ap_distance, move_sequence):
    blocking_penalty = 0
    distance_penalty = 0

    for i in range(len(state)):
        lane = state[i]
        max_priority = 0
        occupied_slots = sum(1 for x in lane if x != 0)

        for j in range(len(lane)):
            if lane[j] != 0:
                if lane[j] < max_priority:
                    blocking_penalty += (6 - lane[j]) * (j + 1) ** 2
                max_priority = lane[j]

    for i in range(len(state)):
        lane = state[i]
        occupied_slots = sum(1 for x in lane if x != 0)
        for j in range(len(lane) - 1):
            if lane[j] != 0 and any(lane[k] < lane[j] for k in range(j + 1, len(lane))):
                distance_penalty += (sum(ap_distance[i][k] * occupied_slots for k in range(len(state)) if any(state[k])) * 0.3)

    backward_blocking_penalty = 0
    for i in range(len(state)):
        lane = state[i]
        max_priority = 0
        for j in reversed(range(len(lane))):
            if lane[j] != 0 and (j == len(lane) - 1 or lane[j + 1] < lane[j]):
                backward_blocking_penalty += (6 - lane[j]) * (len(lane) - j) ** 2

    score = blocking_penalty + distance_penalty + backward_blocking_penalty
    return score

"""

HEURISTIC_CODE_REPAIR = HEURISTIC_CODE_INITIAL

# ----------------------------- Problem Setup -----------------------------
class ProblemState:
    def __init__(self, solution, data, parent_problem_state=None):
        self.solution = solution
        self.data = data
        self.parent_solution = parent_problem_state
        self.tabu_dict = parent_problem_state.tabu_dict if parent_problem_state else {}
        self.add_solution_to_tabu()

    def objective(self) -> float:
        return self.get_loaded_move_time() #+ self.get_unloaded_move_time()

    def get_loaded_move_time(self):
        return sum(self.data.ap_distance[m[0]][m[1]] for m in self.solution)

    def get_unloaded_move_time(self):
        return sum(
            self.data.ap_distance[self.solution[i - 1][1]][self.solution[i][0]]
            for i in range(1, len(self.solution))
        )

    def get_move_number(self):
        return len(self.solution)

    def add_solution_to_tabu(self):
        if self.parent_solution is None:
            return
        for i, move in enumerate(self.solution):
            if None in move:
                self.add_tabu_move(i + 1, tuple(self.parent_solution.solution[i]))

    def add_tabu_move(self, move_index: int, move: tuple):
        if move_index not in self.tabu_dict:
            self.tabu_dict[move_index] = []
        if move not in self.tabu_dict[move_index]:
            self.tabu_dict[move_index].append(move)
        if len(self.tabu_dict[move_index]) > MAX_TABU_LENGTH:
            self.tabu_dict[move_index].pop(0)

# ----------------------------- Initial State -----------------------------
def initial_state(data, paras, instance_seed) -> ProblemState:
    # --- generate heuristic ID ---
    heuristic_id = get_heuristic_id(HEURISTIC_CODE_INITIAL)

    # --- use only filename for folder grouping ---
    eoh_file = os.path.basename(paras.eoh_experiment_file)

    # --- build full storage path ---
    path = get_initial_solution_path(heuristic_id, eoh_file, instance_seed)

    # --- try loading cached solution ---
    cached = load_initial_solution(path)
    if cached is not None:
        print(f"[INIT] Loaded cached initial solution from: {path}")
        return ProblemState([list(mv) for mv in cached], data)

    # --- compute initial solution via heuristic ---
    print("[INIT] No cached initial solution found --> running A* heuristic...")
    heuristic_module = types.ModuleType("heuristic_module")
    exec(HEURISTIC_CODE_INITIAL, heuristic_module.__dict__)

    result = astar_multibay_premarshalling(heuristic_module, data)
    if result["solution"] is None:
        raise Exception("Initial A* failed to find a solution.")

    solution = [list(move) for move in result["solution"]]

    # --- save computed solution ---
    save_initial_solution(path, solution)
    print(f"[INIT] Saved initial solution to: {path}")

    return ProblemState(solution, data)


def apply_scoring(state: ProblemState, scores, k: int):
    """
    Removes the top-k scored positions from 'state.solution' by setting them to None.

    - state.solution: list of [pickup_ap, dropoff_ap]
    - scores: list of [score_pickup, score_dropoff] aligned with state.solution
    - k: number of positions to remove
    Returns only the modified solution list.
    """
    if not scores or k <= 0:
        return state.solution

    # Flatten scores into (row_idx, pos_idx, score)
    flat_scores = []
    for row_idx, row_scores in enumerate(scores):
        for pos_idx, score in enumerate(row_scores):
            #if pos_idx ==0:
                flat_scores.append((row_idx, pos_idx, score))

    # Pick top-k by score
    top_positions = sorted(flat_scores, key=lambda x: x[2], reverse=True)[:k]

    destroyed_solution = copy.deepcopy(state.solution)

    for row_idx, pos_idx, _ in top_positions:
        destroyed_solution[row_idx][pos_idx] = None  # remove that position

    return destroyed_solution


# ----------------------------- Destroy -----------------------------
def destroy(current: ProblemState, rng: rnd.Generator) -> ProblemState:
    destroyed = ProblemState(copy.deepcopy(current.solution), current.data, current)

    scores, k = destroy_scoring(destroyed.solution, destroyed.data, rng)

    destroyed_solution = apply_scoring(destroyed, scores, k)

    return ProblemState(solution=destroyed_solution, data=current.data, parent_problem_state=current)


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

# ----------------------------- WarehouseNode & Repair -----------------------------
class WarehouseNode:
    def __init__(self, warehouse, g, heuristic_fn, parent=None, move=None, move_index=None, problem_state=None):
        self._list_state = None
        self._serial = None
        self.state = warehouse
        self.problem_state = problem_state
        self.move = move
        self.move_index = move_index
        if parent is None:
            self.move_sequence = []
            self.h = 0
        else:
            self.move_sequence = parent.move_sequence + [[move[0], move[1]]] if parent and move is not None else []
            self.h = heuristic_fn.score_state(self.to_list(), warehouse.ap_distance, self.move_sequence)

            if move is not None:
                tabu_list = problem_state.tabu_dict.get(move_index, [])
                if (move[0], move[1]) in tabu_list:
                    self.h += TABU_PENALTY

        self.g = g
        self.f = self.g + self.h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f

    def to_list(self):
        if self._list_state is None:
            self._list_state = convert_vl_to_list(self.state.virtual_lanes)
        return self._list_state

    def serialize(self):
        if self._serial is None:
            lst = self.to_list()
            self._serial = tuple(tuple(stack) for stack in lst)
        return self._serial

    def get_neighbors(self, heuristic_fn, partial_move=None, n_closest=300):
        lanes = self.state.virtual_lanes
        ap_distance = self.state.ap_distance

        load_indices = [i for i, lane in enumerate(lanes) if np.any(lane.stacks != 0)]
        slot_indices = [i for i, lane in enumerate(lanes) if np.any(lane.stacks == 0)]

        neighbors = []
        candidate_moves = set()

        original_move = None
        if self.problem_state.parent_solution is not None:
            if self.move_index < len(self.problem_state.parent_solution.solution):
                original_move = self.problem_state.parent_solution.solution[self.move_index]

        # ============================================================
        # CASE 1: [None, None]
        # ============================================================
        if partial_move == [None, None] and original_move is not None:

            orig_from, orig_to = original_move

            # Original move
            if orig_from in load_indices and orig_to in slot_indices and orig_from != orig_to:
                candidate_moves.add((orig_from, orig_to))

            # orig_from → closest dropoffs
            if orig_from in load_indices:
                closest_dropoffs = sorted(
                    slot_indices,
                    key=lambda t: ap_distance[orig_from][t]
                )[:n_closest]

                for t in closest_dropoffs:
                    if t != orig_from:
                        candidate_moves.add((orig_from, t))

            # closest pickups → orig_to
            if orig_to in slot_indices:
                closest_pickups = sorted(
                    load_indices,
                    key=lambda f: ap_distance[f][orig_to]
                )[:n_closest]

                for f in closest_pickups:
                    if f != orig_to:
                        candidate_moves.add((f, orig_to))

        # ============================================================
        # CASE 2: [fixed_from, None]
        # ============================================================
        elif partial_move and partial_move[0] is not None and partial_move[1] is None:

            fixed_from = partial_move[0]

            if fixed_from in load_indices:

                # Original move if consistent
                if original_move is not None:
                    orig_from, orig_to = original_move
                    if orig_from == fixed_from and orig_to in slot_indices:
                        candidate_moves.add((orig_from, orig_to))

                # Closest dropoffs from fixed_from
                closest_dropoffs = sorted(
                    slot_indices,
                    key=lambda t: ap_distance[fixed_from][t]
                )[:n_closest]

                for t in closest_dropoffs:
                    if t != fixed_from:
                        candidate_moves.add((fixed_from, t))

        # ============================================================
        # CASE 3: [None, fixed_to]
        # ============================================================
        elif partial_move and partial_move[0] is None and partial_move[1] is not None:

            fixed_to = partial_move[1]

            if fixed_to in slot_indices:

                # Original move if consistent
                if original_move is not None:
                    orig_from, orig_to = original_move
                    if orig_to == fixed_to and orig_from in load_indices:
                        candidate_moves.add((orig_from, orig_to))

                # Closest pickups to fixed_to
                closest_pickups = sorted(
                    load_indices,
                    key=lambda f: ap_distance[f][fixed_to]
                )[:n_closest]

                for f in closest_pickups:
                    if f != fixed_to:
                        candidate_moves.add((f, fixed_to))

        # ============================================================
        # CASE 4: Fully fixed move (no branching)
        # ============================================================
        elif partial_move and partial_move[0] is not None and partial_move[1] is not None:

            f, t = partial_move
            if f in load_indices and t in slot_indices and f != t:
                candidate_moves.add((f, t))

        # ============================================================
        # Fallback (safety)
        # ============================================================
        else:
            candidate_moves = {
                (f, t)
                for f in load_indices
                for t in slot_indices
                if f != t
            }

        # ============================================================
        # Generate neighbors
        # ============================================================
        for from_idx, to_idx in candidate_moves:

            # Safe reverse-move pruning
            if self.parent and self.parent.move == (to_idx, from_idx):
                continue

            wh_neighbor = copy.copy(self.state)
            new_lanes = list(lanes)

            new_lane_from, moved_load = lanes[from_idx].remove_load()
            new_lane_to = lanes[to_idx].add_load(moved_load)

            new_lanes[from_idx] = new_lane_from
            new_lanes[to_idx] = new_lane_to
            wh_neighbor.virtual_lanes = new_lanes

            neighbor_node = WarehouseNode(
                wh_neighbor,
                g=self.g + HANDLING_TIME * 2 + ap_distance[from_idx][to_idx],
                heuristic_fn=heuristic_fn,
                parent=self,
                move=(from_idx, to_idx),
                move_index=(self.move_index or 0) + 1,
                problem_state=self.problem_state
            )

            neighbors.append(neighbor_node)

        # Optional debug
        #print(len(neighbors))

        return neighbors

    def is_goal(self):
        for stack in self.to_list():
            for i in range(len(stack) - 1):
                if stack[i] > stack[i + 1]:
                    return False
        return True

    @staticmethod
    def reconstruct_path(node):
        path = []
        while node.parent is not None:
            path.append((node.move, node.state))
            node = node.parent
        return list(reversed(path))

# ----------------------------- Repair -----------------------------
def repair(destroyed: ProblemState, rng: rnd.Generator) -> ProblemState:
    heuristic_module_repair = types.ModuleType("heuristic_module")
    exec(HEURISTIC_CODE_REPAIR, heuristic_module_repair.__dict__)

    def full_astar_repair():
        root = WarehouseNode(
            destroyed.data,
            g=0,
            heuristic_fn=heuristic_module_repair,
            move_index=0,
            problem_state=destroyed
        )
        open_list = [(root.f, 0, 0, root)]
        closed = set()
        evaluated_nodes = 1

        total_start = time.time()
        while open_list:
            _, _, _, node = heapq.heappop(open_list)
            state_id = (node.serialize(), node.move_index)
            if state_id in closed:
                continue
            closed.add(state_id)

            if node.move_index > len(destroyed.solution):
                continue

            if node.is_goal():
                total_time = time.time() - total_start
                path = WarehouseNode.reconstruct_path(node)
                #print(f"Move repair successful! ")
                return {
                    'solution': [list(mv) for mv, _ in path],
                    'solution_time': total_time,
                    'evaluated_nodes': evaluated_nodes,
                }
            if node.move_index < len(destroyed.solution):
                partial_move = destroyed.solution[node.move_index]
            else:
                partial_move = None

            for neighbor in node.get_neighbors(heuristic_module_repair, partial_move=partial_move):
                evaluated_nodes += 1
                heapq.heappush(open_list, (
                neighbor.f, neighbor.state.ap_distance[neighbor.move[0], neighbor.move[1]], evaluated_nodes, neighbor))

                total_time = (time.time() - total_start)
                if total_time > 10:
                    print(f"Move repair timed out: {round(total_time,4)} s - Return parent solution.")
                    return {
                    'solution': destroyed.parent_solution.solution,
                    'solution_time': total_time,
                    'evaluated_nodes': evaluated_nodes,
                }


        raise ValueError("Repair failed: No valid sequence found.")

    result = full_astar_repair()
    new_state = ProblemState(solution=result['solution'], data=destroyed.data, parent_problem_state=destroyed)

    return new_state

# ----------------------------- ALNS Setup -----------------------------
def load_data(instance_seed=0, paras=None):
    eoh_experiment_file = paras.eoh_experiment_file
    configs, _ = load_experiments(eoh_experiment_file)
    config = configs[instance_seed]
    access_directions = get_access_directions(config)
    path = os.path.join(os.getenv("BASE_PATH"), "data", "mr_examples")
    wh = Warehouse(os.path.join(path, config["layout_file"]), access_directions)
    wh.virtual_lanes = create_virtual_lane(config)

    wh.robot_number = int(paras.eoh_experiment_file.split("robots_")[1].split("_")[0].strip(".json"))
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

def stop(num_iters : int, runtime_limit : int, no_improvement_time : int, no_improvement_iters : int):
    return OrStop(NoImprovement(no_improvement_iters), NoImprovementTime(no_improvement_time), MaxIterationsAndMaxRuntime(num_iters, runtime_limit))

def select(num_destroy: int, num_repair: int):
    return RouletteWheel([5, 2, 1, 0.5], 0.8, num_destroy, num_repair)

def accept(init: ProblemState, num_iters : int):
    return RecordToRecordTravel.autofit(
        init.objective(), 0.02, 0, num_iters
    )

def info(_):
    print("Move-Phase: Kein spezieller Info-Output.")

# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    load_dotenv()
    print("Starting ALNS Move Optimization...")
    lns = LNS_Base(problem="lns_multi_robot_upmp_move_search")

    paras_instance = Paras()
    # add eoh_experiment_file to paras
    paras_instance.set_paras(
        eoh_experiment_file="exp_bay_5_wh_2_fill_0.8_seed_0-9_priority_5_robots_2_ad_1010.json",
        lns_paras = {
            "num_iters" : 1000,
            "runtime_limit" : 30,
            "no_improvement_time" : 10,
            "no_improvement_iters" : 1000,
        }
    )

    lns.init_lns(0, paras= paras_instance)

    lns.run()
    lns.get_result()
    lns.plot_objective()
