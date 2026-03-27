

from copy import deepcopy

import numpy as np
import time

from solver.ceoh.methods.selection import prob_rank

from solver.ceoh.llm.models import get_model_info

from solver.ceoh.problems import problems
from solver.ceoh.utils.getParas import Paras

from solver.ceoh.methods.eoh.eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from solver.ceoh.methods.eoh.evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures

import os

import json

import multiprocessing
import logging

from datetime import datetime


class InterfaceEC():
    def __init__(self, paras, prompt_prob, eval_problem, select, idea_select, **kwargs):

        self.detailed_output = os.environ["DETAILED_OUTPUT"] == 'True'

        print(f'Config timout is ignored: {paras.eva_timeout}')

        self.paras = paras

        # LLM settings
        self.pop_size = paras.ec_pop_size
        self.interface_eval = eval_problem
        self.prompts = prompt_prob.prompts

        self.evol = Evolution(paras, self.prompts, **kwargs)

        self.m = paras.ec_m
        self.debug = paras.exp_debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select

        self.n_p = paras.exp_n_proc

        self.idea_select = idea_select

        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator

        self.save_file_folder = os.path.join(os.environ["CURRENT_EXPERIMENT"], 'all_programs')

        self.use_idea = paras.ec_use_idea

        self.ideas = None
        if self.use_idea:
            self.idea_number = paras.idea_number
            try:
                print("Loading scored ideas...")
                file_path_ideas = os.path.join(
                    paras.base_path,
                    "data",
                    "eoh_papers_idea_extraction",
                    paras.problem,
                    paras.llm_model_idea,
                    "scored_ideas",
                    "scored_ideas.json"
                )
                with open(file_path_ideas, "r", encoding="utf-8") as f:
                    scored_ideas = json.load(f)
                self.ideas = scored_ideas if self.use_idea else []

            except FileNotFoundError:
                print("ERROR: scored_ideas.json not found.")
                print("Disable Ideas...")
                self.use_idea = False
                raise FileNotFoundError("scored_ideas.json not found. Please check the file path or disable ideas.")

    def get_offspring_name(self,pop_n, operator, i, time_str):
            return f'pop_{pop_n}_op_{operator}_n{i}_{time_str}_loop{self.paras.current_coupled_iteration}_{self.paras.target_subproblem}'

    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def population_generation(self):
        
        n_create = 2
        
        population = []

        for i in range(n_create):
            _,pop = self.get_algorithm([],self.paras.ec_initial_operator)
            for p in pop:
                population.append(p)
        return population
    
    def population_generation_seed(self,seeds,n_p):

        population = []
        fitness = []

        if self.n_p > 1:
            fitness, detailed_scores = Parallel(n_jobs=self.n_p)(
                delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        else:
            for seed in seeds:
                f, _ = self.interface_eval.evaluate(seed['code'])
                fitness.append(f)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = float(np.round(obj, 5))
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population
    

    def _get_alg(self,pop,operator):

        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }

        if self.use_idea:
            ideas = self.idea_select.parent_selection(self.ideas, self.idea_number)
        else:
            ideas = []

        if operator == "i1":
            parents = None
            [offspring['code'],offspring['algorithm']], prompt, full_res =  self.evol.i1()
        elif operator == "e1":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.e1(parents)
        elif operator == "e2":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.e2(parents)
        elif operator == "m1":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.m1(parents[0])
        elif operator == "m2":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.m2(parents[0])
        elif operator == "m3":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.m3(parents[0])
        elif operator == "p0":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.p0(ideas)
        elif operator == "p1":
            parents = self.select.parent_selection(pop, self.m)
            [offspring['code'],offspring['algorithm']], prompt, full_res = self.evol.p1(parents,ideas)
        elif operator == "i1coupled":
            parents = None
            offspring, prompt, full_res =  self.evol.i1coupled()
        elif operator == "e1coupled":
            parents = self.select.parent_selection(pop,self.m)
            offspring, prompt, full_res =  self.evol.e1coupled(parents)
        elif operator == "e2coupled":
            parents = self.select.parent_selection(pop, self.m)
            offspring, prompt, full_res = self.evol.e2coupled(parents)
        elif operator == "m1coupled":
            parents = self.select.parent_selection(pop, 1)
            offspring, prompt, full_res = self.evol.m1coupled(parents)
        elif operator == "m2coupled":
            parents = self.select.parent_selection(pop, 1)
            offspring, prompt, full_res = self.evol.m2coupled(parents)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        return parents, offspring, prompt, full_res, ideas

    def get_offspring(self, pop, operator, save_file=""):

        possible_exception = ""
        elapsed_time = None
        detailed_scores = None
        code = "FAILED TO EXTRACT JSON"
        prompt = "FAILED TO EXTRACT JSON"
        full_res = "FAILED TO EXTRACT JSON"
        detailed_scores = "FAILED TO EXTRACT JSON"
        ideas = "FAILED TO EXTRACT JSON"

        try:

            p, offspring, prompt, full_res, ideas = self._get_alg(pop, operator)

            if offspring is None:
                print("[EOH_EC]: Offspring is none")
                logging.info("[EOH_EC]: Offspring is none")

            if self.use_numba:
                # Regular expression pattern to match function definitions
                pattern = r"def\s+(\w+)\s*\(.*\):"

                # Search for function definitions in the code
                match = re.search(pattern, offspring['code'])

                function_name = match.group(1)

                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else :
                print("-------- offspring ------")
                print(offspring)
                if isinstance(offspring, dict) and "sub_offsprings" not in offspring:
                    code = offspring['code']
                elif isinstance(offspring["sub_offsprings"], list):
                    coupled_code_dict = {}
                    for o in offspring["sub_offsprings"]:
                        print(o)
                        coupled_code_dict[o["subproblem"]] = o["code"]
                    code = coupled_code_dict
                    print(coupled_code_dict)
            print("-------- CODE ------")
            print(code)

            start_time = time.perf_counter_ns()  # Start timing

            if self.n_p > 1:

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.interface_eval.evaluate, code)
                    fitness, detailed_scores = future.result(timeout=self.timeout)
                    offspring['objective'] = float(np.round(fitness, 5))
                    future.cancel()
            else:
                fitness, detailed_scores = self.interface_eval.evaluate(code)
                print(fitness)
                offspring['objective'] = float(np.round(fitness, 5))
                print("Objective ", offspring['objective'])


            end_time = time.perf_counter_ns()  # End timing
            elapsed_time = (end_time - start_time)  # Calculate elapsed time

            print(f"Thread Time: {elapsed_time}")
            #logging.info(f"Thread Time: {elapsed_time}")


        except Exception as e:

            possible_exception = e
            print(e)
            print("OFFSPRING CREATION FAILED")
            logging.info(f"OFFSPRING CREATION FAILED: {e}")
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None,
                'evaluation_time': None
            }
            p = None

        offspring['evaluation_time'] = elapsed_time  # Store the evaluation time
        offspring['offspring_id']= save_file

        offspring_save = deepcopy(offspring)

        add_exception_string = "" if possible_exception == "" else "_Exception"
        with open(os.path.join(self.save_file_folder, f'program_{save_file}{add_exception_string}.json'), 'w+') as file:
            current_file = {
                'exception': str(possible_exception),
                'p': p,
                'offspring': offspring_save,
                'original_code': code,
                'prompt': prompt,
                'full_request': full_res,
                'details': detailed_scores,
                "ideas": ideas
            }

            try:
                current_file['full_request']["response"] = [line + '\n' for line in
                                                            current_file['full_request']["choices"][0]["message"]["content"].split("\n")]
            except:
                pass

            try:
                current_file['full_request']["response"] = [line + '\n' for line in
                                                            current_file['full_request']["response"].split("\n")]
            except:
                pass

            try:
                if current_file["offspring"]['algorithm'] is not None:
                    current_file["offspring"]['algorithm'] = [line + '\n' for line in current_file["offspring"]['algorithm'].split("\n")]
            except:
                pass
            try:
                if current_file["offspring"]['code'] is not None:
                    current_file["offspring"]['code'] = [line + '\n' for line in current_file["offspring"]['code'].split("\n")]
            except:
                pass

            json.dump(current_file, file, indent=4)

        # Round the objective values
        return p, offspring

    
    def get_algorithm(self, pop, operator, pop_n="0"):
        results = []

        try:
            time_str = datetime.now().strftime("%y%m%d_%H%M%S")
            logging.info(f"Starting parallel execution for population {pop_n} with operator {operator} at {time_str}")

            results = Parallel(n_jobs=self.n_p)(
                delayed(self.get_offspring)(pop, operator, self.get_offspring_name(pop_n, operator, i, time_str)) for i in
                range(self.pop_size)
            )

            if self.detailed_output:
                print('###################### Population Results####################')
                print(results)

        except multiprocessing.TimeoutError:
            logging.error("TIMEOUT ERROR IN PARALLEL: This should not be possible!.")
            print("TIMEOUT ERROR IN PARALLEL: This should not be possible!.")
        except Exception as e:
            logging.error(f"Error: {e}")
            print(f"Error: {e}")
            logging.error("Parallel time out.")
            print("Parallel time out.")

        time.sleep(2)

        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                logging.debug(f">>> check offsprings: \n {off}")
                print(f">>> check offsprings: \n {off}")

        logging.info(f"Finished processing population {pop_n} with operator {operator}.")
        print(f"Finished processing population {pop_n} with operator {operator}.")
        return out_p, out_off


if __name__ == "__main__":

    eoh_experiment_file = (
            "exp_bay_5_wh_2_fill_0.8_seed_0-9_priority_5_robots_2_ad_1010.json"
        )

    os.environ["CURRENT_EXPERIMENT"] =r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling\eoh_results\results_20251218_114155_lns_multi_robot_upmp_coupled_qwen25coder32b_ueFalse_uiFalse_uitypeNone"
    os.environ["OUTPUT_PATH"] = r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling\eoh_results\results_20251218_114155_lns_multi_robot_upmp_coupled_qwen25coder32b_ueFalse_uiFalse_uitypeNone"
    os.environ["LLM_LOCAL_URL"] = "https://api.imi-services.imi.kit.edu/api/generate"
    os.environ["MODEL_NAME"] = 'qwen2.5-coder:32b'  # "gemma3:27b" 'llama3.1:70b', 'gemma2:27b', 'nemotron:latest', 'qwen2.5-coder:32b', 'gpt-4o-2024-08-06'

    llm_api_endpoint, add_url_info, api_key, llm_use_local = get_model_info('qwen2.5-coder:32b')

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
            alns_cycles_iters=5,
            alns_cycles_timeout=60,
            coupled_problems= ["lns_multi_robot_upmp_coupled"],
    )

    paras = Paras()

    paras.set_paras(
            **base_paras,
            method = 'ceoh',
            ec_operators= ['m1coupled'],
            ec_use_math_model =False,
            problem='lns_multi_robot_upmp_coupled',
            target_subproblem='lns_multi_robot_upmp_coupled',
            llm_api_endpoint=llm_api_endpoint,
            llm_api_key="",
            llm_model='qwen2.5-coder:32b',
            llm_use_local=True,
            llm_local_url=llm_api_endpoint,
            llm_temperature=None,
            llm_url_info=add_url_info,
    )


    subProblemGenerator = problems.Probs(paras)
    problem = subProblemGenerator.get_problem()  # problem evaluation function

    interface_ec = InterfaceEC(paras, problem, problem, prob_rank, prob_rank)

    pop = [
     {
          "objective": 598.0,
          "sub_offsprings": [
               {
                    "subproblem": "lns_multi_robot_upmp_move_search",
                    "algorithm": "Score each element of the current solution based on move distance and dependencies. Prioritize removing moves that have higher distances and fewer dependencies to reduce disruption.",
                    "code": "def destroy_scoring(current_solution, data, rng):\n    scores = [[0, 0] for _ in current_solution]\n    k_max = len(current_solution) // 5\n    k = max(1, rng.integers(1, k_max + 1))\n    \n    for i, move in enumerate(current_solution):\n        pickup, dropoff = move\n        if pickup is not None and dropoff is not None:\n            distance_score = data.ap_distance[pickup][dropoff]\n            dependency_count = sum(1 for dep in data.move_dependencies['equal_prio'].values() if i in [pair[0] or pair[1] for pair in dep])\n            scores[i] = [distance_score, -dependency_count]\n    \n    return scores, k"
               },
               {
                    "subproblem": "lns_multi_robot_upmp_vrp",
                    "algorithm": "Score each move based on its start time and dependencies. Prioritize removing moves that are scheduled later and have fewer dependencies to reduce overall makespan disruption.",
                    "code": "def destroy_scoring(current_solution, data, rng):\n    scores = []\n    k_max = sum(len(robot_moves) for robot_moves in current_solution) // 5\n    k = max(1, rng.integers(1, k_max + 1))\n    \n    for robot_moves in current_solution:\n        for move_dict in robot_moves[1:]:  # Skip the starting position (id=0)\n            id_ = move_dict['id']\n            start_time_score = move_dict['start_time']\n            dependency_count = sum(1 for dep in data.move_dependencies['equal_prio'].values() if id_ in [pair[0] or pair[1] for pair in dep])\n            scores.append((id_, (start_time_score, -dependency_count)))\n    \n    return scores, k"
               }
          ],
          "evaluation_time": 28882020100,
          "offspring_id": "pop_0_op_i1coupled_n0_251218_114418_loop0_lns_multi_robot_upmp_coupled"
     },
     {
          "objective": 991.0,
          "sub_offsprings": [
               {
                    "subproblem": "lns_multi_robot_upmp_move_search",
                    "algorithm": "Score each element (access-point) of the current solution based on move frequency, loaded move distance, and dependency constraints to decide which parts to remove.",
                    "code": "def destroy_scoring(current_solution, data, rng):\n    scores = [[0, 0] for _ in current_solution]\n    move_count = [0] * len(data.ap_distance)\n    total_distance = [0] * len(data.ap_distance)\n    \n    for move in current_solution:\n        if move[0] is not None and move[1] is not None:\n            move_count[move[0]] += 1\n            move_count[move[1]] += 1\n            total_distance[move[0]] += data.ap_distance[move[0]][move[1]]\n            total_distance[move[1]] += data.ap_distance[move[0]][move[1]]\n    \n    for i, move in enumerate(current_solution):\n        if move[0] is not None:\n            scores[i][0] = (move_count[move[0]] * 0.5) + total_distance[move[0]]\n        if move[1] is not None:\n            scores[i][1] = (move_count[move[1]] * 0.5) + total_distance[move[1]]\n    \n    k = max(1, min(len(current_solution) // 5, rng.poisson(2)))\n    return scores, k"
               },
               {
                    "subproblem": "lns_multi_robot_upmp_vrp",
                    "algorithm": "Score each move based on its duration, the distance traveled, and dependency constraints to decide which parts to remove.",
                    "code": "def destroy_scoring(current_solution, data, rng):\n    scores = []\n    move_durations = {move['id']: move['end_time'] - move['start_time'] for robot_moves in current_solution for move in robot_moves if move['id'] != 0}\n    \n    for robot_moves in current_solution:\n        for move in robot_moves:\n            if move['id'] == 0:\n                continue\n            move_score = (move_durations[move['id']] * 0.7) + data.ap_distance[move['move'][0]][move['move'][1]]\n            scores.append((move['id'], move_score))\n    \n    k = max(1, min(len(scores) // 5, rng.poisson(2)))\n    return scores, k"
               }
          ],
          "evaluation_time": 20152779100,
          "offspring_id": "pop_0_op_i1coupled_n2_251218_114418_loop0_lns_multi_robot_upmp_coupled"
     },
     {
          "objective": 1079.0,
          "sub_offsprings": [
               {
                    "subproblem": "lns_multi_robot_upmp_move_search",
                    "algorithm": "Score each element based on the distance and dependencies. Higher scores for longer distances or fewer dependencies to break fewer constraints.",
                    "code": "def destroy_scoring(current_solution, data, rng):\n    scores = [[0, 0] for _ in current_solution]\n    for i, move in enumerate(current_solution):\n        pickup, dropoff = move\n        if pickup is not None and dropoff is not None:\n            dist_score = data.ap_distance[pickup][dropoff]\n            dep_score = sum(1 for deps in data.move_dependencies['equal_prio'].values() if i in [dep[0] or dep[1] for dep in deps])\n            scores[i] = [dist_score - dep_score, dist_score - dep_score]\n    k = rng.integers(1, max(2, len(current_solution) // 5))\n    return scores, k"
               },
               {
                    "subproblem": "lns_multi_robot_upmp_vrp",
                    "algorithm": "Score each move based on duration and dependencies. Higher scores for longer durations or fewer dependencies to break fewer constraints.",
                    "code": "def destroy_scoring(current_solution, data, rng):\n    scores = []\n    move_ids = [move['id'] for robot_moves in current_solution for move in robot_moves if move['id'] != 0]\n    for id_ in move_ids:\n        move = next(move for robot_moves in current_solution for move in robot_moves if move['id'] == id_)\n        duration_score = move['end_time'] - move['start_time']\n        dep_score = sum(1 for deps in data.move_dependencies['equal_prio'].values() if id_ in [dep[0] or dep[1] for dep in deps])\n        scores.append((id_, duration_score - dep_score))\n    k = rng.integers(1, max(2, len(scores) // 5))\n    return scores, k"
               }
          ],
          "evaluation_time": 20116226700,
          "offspring_id": "pop_0_op_i1coupled_n2_251218_114201_loop0_lns_multi_robot_upmp_coupled"
     }
]

    out_p, out_off = interface_ec.get_algorithm(pop, "m2coupled",0)

    print(out_p)
    print("___________________")
    print(out_off)