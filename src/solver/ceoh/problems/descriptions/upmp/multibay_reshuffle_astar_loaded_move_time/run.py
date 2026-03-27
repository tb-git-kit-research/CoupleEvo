
import copy
import time
import numpy as np
import types
import warnings
import heapq
import os
import threading
from multiprocessing import Process, Queue
from dotenv import load_dotenv

from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar_loaded_move_time.prompts import GetPrompts
from util.mr_to_ceoh_util import (
    load_experiments,
    get_access_directions,
    create_virtual_lane,
    convert_vl_to_list
)
from problems.multibay_reshuffeling.bay.warehouse import Warehouse

load_dotenv()

MAX_NUMBER_OF_MOVES = 100
TIMEOUT_SECONDS = 60
USE_REFERENCE_SOLUTION = True
MAX_EVALUATED_NODES = 1_000_000
N_WORKERS = 10
HANDLING_TIME = 30

class WarehouseNode:
    def __init__(self, warehouse, g, heuristic_fn, parent=None, move=(0,0)):
        self.state = warehouse
        self.g = g
        if parent is None:
            self.move_sequence = []
            self.h = 0
        else:
            self.move_sequence = parent.move_sequence + [[move[0], move[1]]] if parent and move is not None else []
            self.h = heuristic_fn(self.to_list(), warehouse.ap_distance, self.move_sequence)
        self.f = self.g + self.h
        self.parent = parent
        self.move = move

    def __lt__(self, other):
        return self.f < other.f

    def to_list(self):
        return convert_vl_to_list(self.state.virtual_lanes)

    def serialize(self):
        return tuple(tuple(stack) for stack in self.to_list())

    def is_goal(self):
        for stack in self.to_list():
            for i in range(len(stack) - 1):
                if stack[i] > stack[i + 1]:
                    return False
        return True

    def get_objective_value(self):
        if not self.is_goal():
            return MAX_NUMBER_OF_MOVES * HANDLING_TIME * 2
        path = self.reconstruct_path(self)
        return len(path) * HANDLING_TIME * 2 + self.get_loaded_move_distance(path)

    @staticmethod
    def reconstruct_path(node):
        path = []
        while node.parent is not None:
            path.append((node.move, node.state))
            node = node.parent
        return list(reversed(path))

    @staticmethod
    def get_loaded_move_distance(path):
        distance = 0
        for move, warehouse in path:
            distance += warehouse.ap_distance[move[0], move[1]]
        return distance

    @staticmethod
    def get_unloaded_move_distance(path):
        return sum(path[i][1].ap_distance[path[i - 1][0][1], path[i][0][0]] for i in range(1, len(path))) if len(
                path) > 1 else 0

    def get_neighbors(self, heuristic_fn):
        lanes = self.state.virtual_lanes
        load_indices = [i for i, lane in enumerate(lanes) if np.any(lane.stacks != 0)]
        slot_indices = [i for i, lane in enumerate(lanes) if np.any(lane.stacks == 0)]

        neighbors = []
        for from_idx in load_indices:
            for to_idx in slot_indices:
                if from_idx == to_idx:
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
                    g=self.g + HANDLING_TIME * 2 + self.state.ap_distance[from_idx][to_idx],
                    heuristic_fn=heuristic_fn,
                    parent=self,
                    move=(from_idx, to_idx)
                )
                neighbors.append(neighbor_node)

        return neighbors

def astar_multibay_premarshalling(heuristics, warehouse):
    open_list = []
    visited = set()
    evaluated_nodes = 0

    root = WarehouseNode(warehouse, g=0, heuristic_fn=heuristics.score_state)
    heapq.heappush(open_list, (root.f, 0, evaluated_nodes, root))
    visited.add(root.serialize())
    start_time = time.time()

    while open_list:
        current_time = time.time()
        if (current_time - start_time > TIMEOUT_SECONDS
                or evaluated_nodes > MAX_EVALUATED_NODES):
            return {
                'g_score': None,
                'h_score': None,
                'f_score': None,
                'evaluated_nodes': evaluated_nodes,
                "solution_time": current_time - start_time,
                "solution": None,
                "number_moves": MAX_NUMBER_OF_MOVES,
                "loaded_move_distance": None,
                "unloaded_move_distance": None,
                "objective_value": current_node.get_objective_value()
            }

        _, _, _, current_node = heapq.heappop(open_list)

        if current_node.is_goal():
            path = current_node.reconstruct_path(current_node)
            return {
                'g_score': current_node.g,
                'h_score': current_node.h,
                'f_score': current_node.f,
                'evaluated_nodes': evaluated_nodes,
                "solution_time": time.time() - start_time,
                "solution": [move for move, _ in path],
                "number_moves": len(path),
                "loaded_move_distance": current_node.get_loaded_move_distance(path),
                "unloaded_move_distance": current_node.get_unloaded_move_distance(path),
                "objective_value": current_node.get_objective_value()
            }

        for neighbor in current_node.get_neighbors(heuristics.score_state):
            if neighbor.serialize() in visited:
                continue
            evaluated_nodes += 1
            visited.add(neighbor.serialize())
            move_cost = neighbor.state.ap_distance[neighbor.move[0], neighbor.move[1]]
            heapq.heappush(open_list, (neighbor.f, move_cost, evaluated_nodes, neighbor))


def reshuffle_worker_main(task_queue, result_queue, code_string, path):
    while True:
        job = task_queue.get()
        if job is None:
            break

        config = job
        try:
            heuristic_module = types.ModuleType("heuristic_module")
            exec(code_string, heuristic_module.__dict__)
            access_directions = get_access_directions(config)
            wh = Warehouse(os.path.join(path, config['layout_file']), access_directions)
            wh.virtual_lanes = create_virtual_lane(config)
            results = astar_multibay_premarshalling(heuristic_module, wh)


            result_queue.put({
                'fitness': results["objective_value"],
                'reference': None,
                'results': results
            })
        except Exception as e:
            result_queue.put({'fitness': None, 'reference': None, 'error': str(e)})


class MULTIBAY_RESHUFFLECONST_ASTAR:
    def __init__(self, eoh_experiment_file, code_string=None, paras=None):
        global MAX_NUMBER_OF_MOVES, TIMEOUT_SECONDS, MAX_EVALUATED_NODES

        if paras is not None:
            MAX_NUMBER_OF_MOVES = paras.get('MAX_NUMBER_OF_MOVES', MAX_NUMBER_OF_MOVES)
            TIMEOUT_SECONDS = paras.get('TIMEOUT_SECONDS', TIMEOUT_SECONDS)
            MAX_EVALUATED_NODES = paras.get('MAX_EVALUATED_NODES', MAX_EVALUATED_NODES)

        self.prompts = GetPrompts()
        self.instance_configs, self.ref_scores = load_experiments(eoh_experiment_file)

        if len(self.instance_configs) == 0:
            print("[INSTANCES ERROR]: No Instances available")
            exit(1)

        if code_string is not None:
            self.fitness = self.evaluate(code_string, paras)

    def evaluate(self, code_string, paras=None):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                instance_configs = self.instance_configs
                number_of_exp = len(instance_configs)

                base_path = os.path.join(os.getenv('BASE_PATH'), 'data', 'mr_examples')

                managers = []

                for _ in range(N_WORKERS):
                    task_queue = Queue()
                    result_queue = Queue()
                    p = Process(target=reshuffle_worker_main, args=(task_queue, result_queue, code_string, base_path))
                    p.start()
                    managers.append((p, task_queue, result_queue))

                results = [None] * number_of_exp
                buckets = [[] for _ in range(N_WORKERS)]

                for i, config in enumerate(instance_configs):
                    buckets[i % N_WORKERS].append((i, config))

                threads = []

                def run_manager(index, jobs):
                    process, task_queue, result_queue = managers[index]
                    for i, config in jobs:
                        task_queue.put(config)
                        start_time = time.time()
                        result = None

                        while (time.time() - start_time) < TIMEOUT_SECONDS + 5:
                            try:
                                candidate = result_queue.get(timeout=0.1)
                                if candidate is not None:
                                    result = candidate
                                    break
                            except Exception:
                                pass

                        results[i] = result
                    task_queue.put(None)

                    # Wait a little for graceful shutdown
                    process.join(timeout=0.1)

                    # If still alive (e.g., heuristic stuck in infinite loop), kill it
                    if process.is_alive():
                        print(f"[WARN] Worker {index} did not exit, force killing.")
                        process.terminate()
                        process.join()

                for i in range(N_WORKERS):
                    t = threading.Thread(target=run_manager, args=(i, buckets[i]))
                    t.start()
                    threads.append(t)

                for t in threads:
                    t.join()

                detailed_fitness = []
                references = []

                algo_moves = []
                algo_loaded_distances = []
                algo_unloaded_distances = []
                algo_objectives = []

                solution_times = []
                evaluated_nodes = []
                solutions = []

                for r, result in enumerate(results):
                    #if result.get("fitness") is None:
                    #    print("[ERROR] Worker error.")

                    detailed_fitness.append(result['fitness'])
                    references.append(result['reference'])

                    algo_moves.append(result["results"]['number_moves'])
                    algo_loaded_distances.append(result["results"]['loaded_move_distance'])
                    algo_unloaded_distances.append(result["results"]['unloaded_move_distance'])
                    algo_objectives.append(result["results"]['objective_value'])

                    solutions.append(result["results"]['solution'])

                    solution_times.append(result["results"]['solution_time'])
                    evaluated_nodes.append(result["results"]['evaluated_nodes'])

                overall_score = sum(detailed_fitness)
                details = {
                    'detailed_fitness': detailed_fitness,
                    'references': references,

                    "move_numbers": algo_moves,
                    "loaded_distances": algo_loaded_distances,
                    "unloaded_distances": algo_unloaded_distances,
                    "objectives": algo_objectives,

                    "solutions": solutions,

                    "solution_time": solution_times,
                    "evaluated_nodes": evaluated_nodes,
                }

                print(f'Overall fitness: {overall_score / number_of_exp}', flush=True)
                return (overall_score / number_of_exp), details

        except Warning as w:
            print(f"[ERROR] A warning was raised and treated as an error: {w}")
            return None
        except Exception as e:
            print(f"[ERROR] An exception occurred: {e}")
            return None


if __name__ == "__main__":
    code_string = """
def score_state(state, ap_distance, move_sequence):
    blocking_penalty = 0
    distance_penalty = 0
    mixed_class_penalty = 0
    deep_blocking_penalty = 0
    non_direct_adjacent_move_penalty = 0

    blocking_count_per_lane = [0] * len(state)

    for i in range(len(state)):
        lane = state[i]
        max_priority = 0
        occupied_slots = sum(1 for x in lane if x != 0)
        unique_classes = set()

        for j in range(len(lane)):
            if lane[j] != 0:
                unique_classes.add(lane[j])
                if lane[j] < max_priority:
                    blocking_penalty += (5 - lane[j]) * (j + 1) ** 1.5
                    deep_blocking_penalty += (5 - lane[j]) * j ** 2.5 / len(lane)
                    blocking_count_per_lane[i] += 1
                max_priority = lane[j]

        if len(unique_classes) > 1:
            decay_factor = 0.9 ** len(unique_classes)
            mixed_class_penalty += sum(4 * x * decay_factor for x in unique_classes)

    for i in range(len(state)):
        lane = state[i]
        occupied_slots = sum(1 for x in lane if x != 0)
        for j in range(len(lane) - 1):
            if lane[j] != 0 and any(lane[k] < lane[j] for k in range(j + 1, len(lane))):
                distance_penalty += sum(ap_distance[i][k] * occupied_slots ** 0.7
                                        for k in range(len(state)) if any(state[k])) * 0.5

    most_blocked_lane = max(range(len(state)), key=lambda i: blocking_count_per_lane[i])

    for move in move_sequence:
        from_idx, to_idx = move
        if not (from_idx == most_blocked_lane or
                to_idx == most_blocked_lane or
                abs(from_idx - most_blocked_lane) == 1 or
                abs(to_idx - most_blocked_lane) == 1):
            non_direct_adjacent_move_penalty += 4

    score = (blocking_penalty
             + distance_penalty
             + mixed_class_penalty
             + deep_blocking_penalty
             + non_direct_adjacent_move_penalty)

    return score

"""
    start_time = time.time()
    eoh_experiment_file = "exp_bay5_wh_2_fill_0.6.json"
    reshuffle_const = MULTIBAY_RESHUFFLECONST_ASTAR(eoh_experiment_file, code_string)
    end_time = time.time()

    if reshuffle_const.fitness is not None:
        print("Fitness Score:", reshuffle_const.fitness)
    else:
        print("Evaluation failed due to an error in the provided heuristic code.")

    print("Total Execution Time:", end_time - start_time, "seconds")