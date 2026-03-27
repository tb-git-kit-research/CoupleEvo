import subprocess
from copy import deepcopy
import time

from dotenv import load_dotenv

from libraries.lns.base_class import LNS_Base
from libraries.lns.wrapper.lns_multi_robot_upmp_move_search import ProblemState as ProblemStateMove
from libraries.lns.wrapper.lns_multi_robot_upmp_vrp import ProblemState as ProblemStateRoute
from solver.ceoh.problems.descriptions.upmp.multibay_reshuffle_astar_loaded_move_time.run import HANDLING_TIME

from solver.ceoh.utils.getParas import Paras
import numpy as np

PLOTTING = False
PRINTOUT = False

def move_to_routing(move_solution, data, prev_routes=None, prev_ids=None):
    """
    Update a Routing solution based on a new move_solution.

    - move_solution: list of updated [from, to] moves (ordered like prev_ids).
    - prev_routes: previous routing solution (with robot assignments).
    - prev_ids: list mapping index -> move id (from old solution).
    """
    number_robots = len(data.robot_aps)
    move_id = 1

    if prev_routes is None or prev_ids is None:
        # --- Build naive routing solution (like before) ---
        routes = []
        for r in range(number_robots):
            route = [{
                "id": 0,
                "move": [data.robot_aps[r], data.robot_aps[r]],
                "start_time": 0,
                "end_time": 0,
                "robot": r,
            }]
            if r == 0:
                for mv in move_solution:
                    route.append({
                        "id": move_id,
                        "move": mv,
                        "start_time": 0,
                        "end_time": 0,
                        "robot": r,
                    })
                    move_id += 1
            routes.append(route)

        # --- Step 1: baseline timing on each route ---
        for route in routes:
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

    else:
        new_tour = []
        for route in prev_routes:
            new_route = []
            for move in route:

                if move["id"] == 0:
                    new_route.append(move)

                    continue
                id = prev_ids.index(move["id"])

                if len(move_solution) <= id:
                    #print(move, "removed")
                    continue


                if move["move"] == move_solution[id]:
                    new_route.append(move)
                else:
                    #print("Move was modified ", move["move"] , move_solution[id])
                    m = {'id':  move["id"],
                         'move': move_solution[id],
                         'start_time':  move["start_time"],
                         'end_time':  move["end_time"],
                         'robot': move["robot"]
                         }
                    new_route.append(m)
            new_tour.append(new_route)
        routes = new_tour

    return routes

def routing_to_move(routing_solution, dependencies):
    """
    Konvertiere eine Routing-Lösung (Liste von Routen mit Dicts)
    in eine Move-Lösung ([[from,to], ...]).
    Sortiert nach start_time, bei Gleichstand nach end_time.
    Beachtet auch Abhängigkeitsregeln für überlappende Moves.
    """
    moves = []

    # Collect all moves except dummy ones
    for route in routing_solution:
        for m in route:
            if isinstance(m, dict) and m.get("id", 0) != 0:
                moves.append({
                    "id": m["id"],
                    "move": list(m["move"]),
                    "start_time": float(m.get("start_time", 0)),
                    "end_time": float(m.get("end_time", 0)),
                })

    # Initial sort by start_time, then end_time
    moves.sort(key=lambda x: (x["start_time"]))

    # ---- Overlap & dependency adjustment ----
    for i in range(len(moves)):
        for j in range(len(moves)):
            start1, end1 = moves[i]["start_time"], moves[i]["end_time"]
            start2, end2 = moves[j]["start_time"], moves[j]["end_time"]

            # Check if they overlap
            if (end1 >= start2 and start1 <= end2):
                move_pair = [moves[i]["id"], moves[j]["id"]]

                # Rule: "start-end" --> i must come before j
                if move_pair in dependencies["unequal_prio"]["start-end"]:
                    if i > j:
                        moves[i], moves[j] = moves[j], moves[i]

                # Rule: "end-start" --> i must come after j
                if move_pair in dependencies["unequal_prio"]["end-start"]:
                    if i < j:
                        moves[i], moves[j] = moves[j], moves[i]

    # Only return the [from, to] move list
    sorted_moves = [m["move"] for m in moves]
    sorted_ids = [m["id"] for m in moves]
    return sorted_moves, sorted_ids


def run_alternating_alns(cycles: int = 100,
                         paras_move=None,
                         paras_route=None,
                         instance_seed=0,
                         lns_move=None,
                         lns_route=None,
                         eoh_experiment_file=None,
                         timeout=60):

    print("Starting Alternating ALNS Optimization")

    print("INITIALIZING MOVE SEARCH STEP")
    if lns_move is None:
        lns_move = LNS_Base(problem="lns_multi_robot_upmp_move_search")
    lns_move.init_lns(instance_seed, paras=paras_move)
    current_state = lns_move._init_sol

    print("INITIALIZING ROUTING STEP \n")
    if lns_route is None:
        lns_route = LNS_Base(problem="lns_multi_robot_upmp_vrp")
    lns_route.init_lns(instance_seed, paras=paras_route)

    routing_solution = None
    astar_solution = None
    sorted_ids = None

    lns_route._init_sol = ProblemStateRoute(
        solution=move_to_routing(
            current_state.solution,
            current_state.data,
            prev_routes=routing_solution,
            prev_ids=sorted_ids,
        ),
        data=current_state.data,
        parent_problem_state=None,
        astar_solution=astar_solution
    )

    objective_history = [lns_route._init_sol.objective()]
    loaded_distance_history = [lns_route._init_sol.get_loaded_move_time()]
    unloaded_distance_history = [lns_route._init_sol.get_unloaded_move_time()]
    move_number_history = [lns_route._init_sol.get_move_number()]
    state_history = [lns_route._init_sol]
    solution_history = [lns_route._init_sol.solution]
    changed_moves_history = []

    current_state = lns_route._init_sol #lns_route._result.best_state
    routing_solution = deepcopy(current_state.solution)

    route_times = []
    move_times = []
    cycle_times = []

    overall_time = 0

    if PRINTOUT:
        print("START CYCLE")
    for cycle in range(cycles):
        cycle_start = time.time()   # <<< start full cycle timer

        # ---------------- MOVE PHASE ----------------
        if PRINTOUT:
            print(f"\n CYCLE {cycle + 1} — MOVE PHASE")
            print(f"MOVE SEARCH PHASE")
        move_time_start = time.time()

        sorted_solution, sorted_ids = routing_to_move(
            current_state.solution,
            current_state.data.move_dependencies
        )

        lns_move._init_sol = ProblemStateMove(
            solution=sorted_solution,
            data=current_state.data
        )

        lns_move.init_lns(instance_seed, paras=paras_move)
        lns_move.run()
        current_state = lns_move._result.best_state

        astar_solution = [[i, m] for i, m in zip(sorted_ids, current_state.solution)]

        move_times.append(time.time() - move_time_start)

        if overall_time + time.time() - cycle_start > timeout:
            if PRINTOUT:
                print("Timout in LNS Break cycle")
            break

        # ---------------- ROUTE PHASE ----------------
        if PRINTOUT:
            print(f"VEHICLE ROUTING PHASE")
        route_time_start = time.time()

        lns_route._init_sol = ProblemStateRoute(
            solution=move_to_routing(
                current_state.solution,
                current_state.data,
                prev_routes=routing_solution,
                prev_ids=sorted_ids,
            ),
            data=current_state.data,
            parent_problem_state=None,
            astar_solution=astar_solution
        )

        lns_route.init_lns(instance_seed, paras=paras_route)
        lns_route.run()
        current_state = lns_route._result.best_state
        if PRINTOUT:
            print("Objective value ", current_state.objective())

        routing_solution = deepcopy(current_state.solution)

        objective_history.append(current_state.objective())
        loaded_distance_history.append(current_state.get_loaded_move_time())
        unloaded_distance_history.append(current_state.get_unloaded_move_time())
        move_number_history.append(current_state.get_move_number())
        state_history.append(current_state)
        solution_history.append(current_state.solution)
        changed_moves_history.append(count_removed_moves(state_history[-2],state_history[-1]))
        route_times.append(time.time() - route_time_start)
        cycle_times.append(time.time() - cycle_start)

        overall_time += time.time() - cycle_start

        if overall_time > timeout:
            print("Timout in LNS Break cycle")
            break

    # ---------------- FINAL OUTPUT ----------------
    if PRINTOUT:
        print(f"\n✅ Alternating Optimization Complete for seed {instance_seed}")
        print(f"Initial Number Moves: {move_number_history[1]:.2f}")
        print(f"Final Number Moves: {move_number_history[-1]:.2f}")

        print(f"Initial Objective: {objective_history[1]:.2f}")
        print(f"Final Objective: {objective_history[-1]:.2f}")

        print(f"Total time all cycles:   {sum(cycle_times):.3f} s")

        print("Initial Solution:", [[item['id'] for item in robot] for robot in state_history[1].solution])
        print("Final Solution:", [[item['id'] for item in robot] for robot in state_history[-1].solution])

        print("Initial Sequence:", [[item['move'] for item in robot] for robot in state_history[0].solution])
        print("Final Sequence:", [[item['move'] for item in robot] for robot in state_history[-1].solution])

        print("Number Changed Moves: ", count_removed_moves(state_history[0], state_history[-1]))

    if PLOTTING:
        plot_objective_graph(
            eoh_experiment_file, instance_seed, loaded_distance_history,
            move_number_history, objective_history, unloaded_distance_history
        )

    return {
        "objective_history" : objective_history,
        "loaded_distance_history" : loaded_distance_history,
        "unloaded_distance_history" : unloaded_distance_history,
        "move_number_history" : move_number_history,
        "cycle_times": cycle_times,
        "solution_history" : solution_history,
        "changed_moves_history" : changed_moves_history,
    }


def plot_objective_graph(eoh_experiment_file, instance_seed, loaded_distance_history, move_number_history,
                         objective_history, unloaded_distance_history):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    # --- Primary axis (Objective + Number of Moves) ---
    ax.plot(objective_history, color='gray', alpha=1, label="Objective Value")
    #ax.bar(range(len(move_number_history)), move_number_history, color='orange', alpha=0.6, label='Number Moves')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    ax.grid(True)
    # --- Secondary axis for Loaded / Unloaded Move Times ---
    ax2 = ax.twinx()
    # ***** Add Total Move Time curve *****
    total_move_time = [
        l + u for l, u in zip(loaded_distance_history, unloaded_distance_history)
    ]
    ax2.plot(total_move_time, color='blue', linestyle='--', marker='s',
            label='Total Move Time')
    ax2.plot(loaded_distance_history, color='green', marker='o',
             label='Loaded Move Time')
    ax2.plot(unloaded_distance_history, color='purple', marker='x',
             label='Unloaded Move Time')
    ax2.set_ylabel("Time")
    # --- Combined Legend ---
    # Get handles/labels from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles1 + handles2, labels1 + labels2, loc='best')
    try:
        plt.title(f"{eoh_experiment_file.strip('.json')} | seed {instance_seed}")
    except:
        plt.title(f" seed {instance_seed}")
    plt.tight_layout()
    plt.show()


from collections import Counter

def count_removed_moves(initial_state, final_state):
    initial_seq = [[item['move'] for item in robot] for robot in initial_state.solution]
    final_seq = [[item['move'] for item in robot] for robot in final_state.solution]

    init_moves  = [tuple(m) for robot in initial_seq for m in robot]
    final_moves = [tuple(m) for robot in final_seq   for m in robot]

    init_count  = Counter(init_moves)
    final_count = Counter(final_moves)

    removed = 0
    for move, count_init in init_count.items():
        count_final = final_count.get(move, 0)
        if count_final < count_init:
            removed += count_init - count_final

    return removed

if __name__ == "__main__":
    subprocess.run(["pip", "install", "."], stdout=subprocess.DEVNULL)
    load_dotenv()
    results_list = []

    instance_seed = 2
    seeds = range(0,1)
    for instance_seed in seeds:
        eoh_experiment_file = "exp_bay_5_wh_2_fill_0.8_seed_0-9_priority_5_robots_2_ad_1010.json"

        paras_move = Paras()
        # add eoh_experiment_file to paras
        paras_move.set_paras(
            eoh_experiment_file=eoh_experiment_file,
            lns_paras = {
                "num_iters" : 10,
                "runtime_limit" : 10,
                "no_improvement_time" : 10,
                "no_improvement_iters" : 10,
            }
        )

        paras_route = Paras()
        # add eoh_experiment_file to paras
        paras_route.set_paras(
            eoh_experiment_file=eoh_experiment_file,
            lns_paras = {
                "num_iters" : 90,
                "runtime_limit" : 10,
                "no_improvement_time" : 10,
                "no_improvement_iters" : 90,
            }
        )
        results_dict = run_alternating_alns(cycles=1000,
                                            paras_move = paras_move,
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

