# Copyright (c) 2024 - Fei Liu (fliu36-c@my.cityu.edu.hk)
#
# Modified by: 
#           Thomas Bömer (thomas.bömer@tu-dortmund.de)
#           Nico Koltermann (nico.koltermann@tu-dortmund.de) 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import shutil
from copy import deepcopy

import numpy as np
import random
import time

from solver.ceoh.utils.getParas import Paras

from .eoh_interface_EC import InterfaceEC
from solver.ceoh.utils.visualizeResults import *
from solver.ceoh.utils.visualizePromptFrequency import *
from solver.ceoh.problems import problems

# main class for eoh
class EOH:

    # initilization
    def __init__(self, paras, problem, select, idea_select, manage, **kwargs):

        self.paras = paras

        self.prob = problem
        self.select = select
        self.idea_select = idea_select
        self.manage = manage

        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model
        self.llm_temperature = paras.llm_temperature

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # assert isinstance(self.use_local_llm, bool)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings
        self.pop_size = paras.ec_pop_size  # population size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_pop_nr
        self.exp_continue_folder = paras.exp_continue_folder

        self.output_path = paras.exp_output_path

        try:
            self.result_folder_name = os.environ["CURRENT_EXPERIMENT"]
        except:
            self.result_folder_name = None

        self.exp_n_proc = paras.exp_n_proc
        
        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        self.llm_url_info = paras.llm_url_info

        # Use Context with EoH
        self.use_example = paras.ec_use_example

        # Parameter for Ideas integration
        self.ideas_iteration_start = paras.ideas_iteration_start
        self.injection_loops = paras.injection_loops

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)

    def copy_files(self):
        """
        Copies JSON files from subdirectories if their generation number is
        less than or equal to `self.load_pop_id`.
        """
        base_path = os.path.join(self.output_path, self.exp_continue_folder)
        target_path = self.result_folder_name
        subfolders = ["pops", "pops_best", "all_programs"]

        for subfolder in subfolders:
            source_dir = os.path.join(base_path, subfolder)
            dest_dir = os.path.join(target_path, subfolder)

            os.makedirs(dest_dir, exist_ok=True)  # Ensure destination exists

            for filename in os.listdir(source_dir):
                if filename.endswith(".json"):
                    try:
                        # Extract generation number based on naming pattern
                        gen_index = -1 if subfolder in ["pops", "pops_best"] else 2
                        gen = int(filename.split("_")[gen_index].split(".")[0])

                        if gen <= int(self.load_pop_id):
                            shutil.copy(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))
                    except (ValueError, IndexError):
                        print(f"Skipping file with unexpected format: {filename}")


    def get_pop_name(self, pop, loop = 0):
        return f'population_generation_{pop}_loop{loop}_{self.paras.target_subproblem}'


    # run eoh
    def run(self):

        print("- Evolution Start -")

        time_start = time.time()

        # Initialize a problem for each subproblem to access prompts.
        problems_to_iterate = []
        for problem_name in self.paras.coupled_problems:
            sub_paras = Paras()
            sub_paras.set_paras(problem = problem_name)
            sub_paras.set_paras(eoh_experiment_file = self.paras.eoh_experiment_file)
            subProblemGenerator = problems.Probs(sub_paras)
            sub_problem = subProblemGenerator.get_problem()  # problem evaluation function
            problems_to_iterate.append(sub_problem)
        eval_problem = self.prob

        # Set up seed heuristics
        if len(self.paras.coupled_problems) >1:
            human_seed_population = []
            for s, sub_problem in enumerate(self.paras.coupled_problems):
                paras = deepcopy(self.paras)
                # add eoh_experiment_file to paras
                paras.set_paras(
                    target_subproblem=sub_problem,
                )

                avg_fitness, results = eval_problem.evaluate(problems_to_iterate[s].human_code_string)
                print(sub_problem, avg_fitness)
                seed_alg = {
                    'algorithm': "",
                    'code': problems_to_iterate[s].human_code_string,
                    'objective': avg_fitness,
                    'other_inf': None,
                    "offspring_id": f"human_{sub_problem}"
                }
                with open(os.path.join(os.environ["CURRENT_EXPERIMENT"], 'all_programs', f'program_human_{sub_problem}.json'), 'w+') as file:
                    current_file = {
                        'offspring': seed_alg,
                        'details': results,
                    }
                    json.dump(current_file, file, indent=4)
                human_seed_population.append(seed_alg)

        if len(self.paras.coupled_problems) == 1 and "coupled" in self.paras.problem:
            subproblems = eval_problem.get_subproblems()

            human_seed_population = []

            paras = deepcopy(self.paras)
            # add eoh_experiment_file to paras
            paras.set_paras(
                    target_subproblem=self.paras.problem,
                )
            try:
                code_string = {
                    subproblems[0]: f"{eval_problem.human_code_string[0]}",
                    subproblems[1]: f"{eval_problem.human_code_string[1]}",
                }
            except IndexError:
                print("Human code strings not defined for: ", eval_problem.problem)

            avg_fitness, results = eval_problem.evaluate(code_string)
            print(paras.problem, avg_fitness)

            offspring = {
                "objective": avg_fitness,
                "sub_offsprings": [
                    {
                        'subproblem': subproblems[0],
                        'algorithm': "",
                        'code': eval_problem.human_code_string[0],
                    },
                    {
                        'subproblem': subproblems[1],
                        'algorithm': "",
                        'code': eval_problem.human_code_string[1],
                    }
                    ],
                'offspring_id': f"human_{self.paras.problem}",
                }
            with open(os.path.join(os.environ["CURRENT_EXPERIMENT"], 'all_programs', f'program_human_{paras.problem}.json'), 'w+') as file:
                current_file = {
                    'offspring': offspring,
                    'details': results,
                }
                json.dump(current_file, file, indent=4)
            human_seed_population.append(offspring)

        for loop in range(self.paras.coupled_iterations):
            print(f"### Starting loop {loop} ###")
            self.paras.set_paras(current_coupled_iteration = loop)
            for s, subproblem in enumerate(problems_to_iterate):
                print("Processing subproblem: ",self.paras.coupled_problems[s])
                self.paras.set_paras(target_subproblem = self.paras.coupled_problems[s])
                prompt_problem = subproblem

                # interface for ec operators
                interface_ec = InterfaceEC(self.paras, prompt_problem, eval_problem, self.select, self.idea_select)

                # initialization
                population = []
                if loop == 0:
                    if self.use_seed:
                        with open(self.seed_path) as file:
                            data = json.load(file)
                        population = interface_ec.population_generation_seed(data,self.exp_n_proc)
                        filename = self.result_folder_name + "/pops/"+self.get_pop_name(0, 0)+".json"
                        with open(filename, 'w+') as f:
                            json.dump(population, f, indent=5)
                        n_start = 0
                    else:
                        if self.load_pop:  # load population from files
                            print("load initial population from " + self.paras.exp_continue_path)
                            with open(self.paras.exp_continue_path) as file:
                                data = json.load(file)
                            for individual in data:
                                population.append(individual)
                            print("initial population has been loaded!")

                            n_start = int(self.load_pop_id) + 1

                            # copy all files from the previous experiment
                            self.copy_files()

                        else:  # create new population
                            print("creating initial population:")
                            population = interface_ec.population_generation()

                            if len(self.paras.coupled_problems) > 1:
                                population.append(human_seed_population[s])
                            if  len(self.paras.coupled_problems) == 1 and "coupled" in self.paras.problem:
                                population.append(human_seed_population[0])


                            population = self.manage.population_management(population, self.pop_size)

                            print(f"Pop initial: ")
                            for off in population:
                                print(" Obj: ", off['objective'], end="|")
                            print()
                            print("initial population has been created!")
                            # Save population to a file
                            filename = self.result_folder_name + "/pops/"+self.get_pop_name(0, 0)+".json"
                            with open(filename, 'w+') as f:
                                json.dump(population, f, indent=5)
                            n_start = 0
                else:
                    print("Try to read data from prior loop. ")
                    with open(self.result_folder_name + "/pops/" + self.get_pop_name(self.n_pop, loop-1) + ".json") as file:
                        data = json.load(file)
                    for individual in data:
                        population.append(individual)
                    os.makedirs(self.result_folder_name + "/transfer", exist_ok=True)
                    filename = self.result_folder_name + "/transfer/" + self.paras.target_subproblem+ "_" +str(loop) + ".json"
                    with open(filename, 'w+') as f:
                        json.dump(population, f, indent=5)
                    n_start = 0
                    print("Read data from ", self.result_folder_name + "/pops/" + self.get_pop_name(self.n_pop-1, self.paras.current_coupled_iteration-1) + ".json")

                # Save the best one to a file
                filename = self.result_folder_name + "/pops_best/" + self.get_pop_name(str(0), loop) + ".json"
                with open(filename, 'w+') as f:
                    json.dump(population[0], f, indent=5)

                # main loop
                n_op = len(self.operators)

                for pop in range(n_start, self.n_pop, 1):
                    print(pop)

                    enable_idea_operator = pop < n_start + self.injection_loops
                    use_idea_in_current_pop = self.ideas_iteration_start < pop

                    if use_idea_in_current_pop and enable_idea_operator:
                        for i in range(n_op):
                            print("Paper idea extraction.")
                            op = "p1"
                            parents, offsprings = interface_ec.get_algorithm(population, op, pop)
                            self.add2pop(population, offsprings)
                            size_act = min(len(population), self.pop_size)
                            population = self.manage.population_management(population, size_act)
                            print()

                    else:
                        for i in range(n_op):
                            op = self.operators[i]
                            print(f" OP: {op}, [{i + 1} / {n_op}] ", end="| \n")
                            op_w = self.operator_weights[i]
                            if (np.random.rand() < op_w):
                                parents, offsprings = interface_ec.get_algorithm(population, op, pop)
                            self.add2pop(population, offsprings)  # Check duplication, and add the new offspring
                            for off in offsprings:
                                print(" Obj: ", off['objective'], end="|")

                            size_act = min(len(population), self.pop_size)
                            population = self.manage.population_management(population, size_act)
                            print()


                    # Save population to a file
                    filename = self.result_folder_name + "/pops/"+ self.get_pop_name(str(pop + 1), loop) + ".json"
                    with open(filename, 'w+') as f:
                        json.dump(population, f, indent=5)

                    # Save the best one to a file
                    filename = self.result_folder_name + "/pops_best/" + self.get_pop_name(str(pop + 1), loop) +  ".json"
                    with open(filename, 'w+') as f:
                        json.dump(population[0], f, indent=5)


                    print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
                    print("Pop Objs: ", end=" ")
                    for i in range(len(population)):
                        print(str(population[i]['objective']) + " ", end="")
                    print()
                    try:
                        load_and_plot_objective_ranges(self.result_folder_name)
                    except Exception as e:
                        print(f"Error plotting objective ranges: {e}")
                    try:
                        plot_evaluation_time(self.result_folder_name)
                    except Exception as e:
                        print(f"Error plotting evaluation time: {e}")

