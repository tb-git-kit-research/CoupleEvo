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

import os

import re
import json
import shutil

from datetime import datetime

def sanitize_name(name: str) -> str:
    """
    Replace invalid characters in a folder name with spaces or underscores.
    Adjust as needed for your OS or naming conventions.
    """
    return re.sub(r'[^A-Za-z0-9.]+', '_', name)

def create_folders(paras):
    # Specify the path where you want to create the folder

    problem = os.getenv('EOH_PROBLEM')
    model = os.getenv('MODEL_NAME')
    model = re.sub('[^A-Za-z0-9]+', '', model)
    print(paras.idea_type)

    idea_type = "None"
    if paras.idea_type == 'injection':
        idea_type = f'INJ'
    if paras.idea_type == f'strategy':
        idea_type = f'STR-{str(paras.idea_number)}'
    if paras.idea_type == 'strategy_e2':
        idea_type = f'STR_E2'

    current = f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{problem}_{model}_ue{paras.ec_use_example}_ui{paras.ec_use_idea}_uitype{idea_type}'
    folder_path = os.path.join(paras.exp_output_path, current)

    print(f'Folder Path: {folder_path}')

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Create the main folder "results"
        os.makedirs(folder_path)

    # Create subfolders inside "results"
    subfolders = ["history", "pops", "pops_best", "all_programs", "visualization"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    os.environ["CURRENT_EXPERIMENT"] = folder_path

def check_instances(paras):

    current_folder = os.environ["CURRENT_EXPERIMENT"]

    # TODO: check eval instances here
    if "reshuffle" in paras.problem:
        experiment_path = os.path.join(os.getenv('BASE_PATH'), "data", "eoh_experiment_config")

        experiments = []
        if paras.eoh_experiment_file != "":
            print(paras.eoh_experiment_file)
            f_exp = open(os.path.join(experiment_path, paras.eoh_experiment_file))
            experiments.append(json.load(f_exp))

        with open(os.path.join(current_folder, "evaluation_experiments.json"), 'w+') as f:
            json.dump('{ experiments: ' + str(experiments) + ' }', f, indent=5)
    elif "vrp" in paras.problem:
        # check folder problem exists vrp_instances/cvrp/test/50.npz
        instance_size = paras.eoh_experiment_file.split("/")[-1].split(".")[0]
        if os.path.join(os.getenv('BASE_PATH'), "vrp_instances", paras.problem, "test", instance_size, ".npz"):
            print("Problem folder with test instances exists")
        else:
            print("Problem folder with test instances does not exist for problem: ", paras.problem, " and instance: ",
                  instance_size)
            exit(1)
    elif "puzzle" in paras.problem:
        # check folder problem exists puzzle_instances
        instance_size = paras.eoh_experiment_file.split("/")[-1].split(".")[0]
        if os.path.join(os.getenv('BASE_PATH'), "puzzle_instances", ".json"):
            print("Problem folder with test instances exists")
        else:
            print("Problem folder with test instances does not exist for problem: ", paras.problem, " and instance: ",
                  instance_size)
            exit(1)

def create_idea_folders(paras):

    current_folder = os.environ["CURRENT_EXPERIMENT"]


    if paras.ec_use_idea:
        try:
            idea_model_name = os.environ["IDEA_MODEL_NAME"]
        except:
            pass

        if idea_model_name == None:
            print("No Ideas Model given...")
            print(f"[IDEAS] Using {paras.llm_model} then.")
            idea_model_name = paras.llm_model
            os.environ["IDEA_MODEL_NAME"] = paras.llm_model

        # save idea_file to experiment folder
        file_path_ideas = os.path.join(os.environ["BASE_PATH"], "data", "eoh_papers_idea_extraction", paras.problem,
                                       idea_model_name, "scored_ideas", "scored_ideas.json")

        if os.path.exists(file_path_ideas) and paras.ui_experiment == 'injection':
            # copy the file to the experiment folder
            shutil.copy(file_path_ideas, os.path.join(current_folder, "scored_ideas.json"))

def save_paras(paras):

    current_folder = os.environ["CURRENT_EXPERIMENT"]

    paras_json = vars(paras).copy()
    paras_json['llm_api_key'] = "---"
    with open(os.path.join(current_folder, "setup.json"), 'w+') as f:
        json.dump(paras_json, f, indent=5)

import os
import shutil

def save_lns_script(file_path):
    current_folder = os.environ["CURRENT_EXPERIMENT"]

    # Ensure the target directory exists
    os.makedirs(current_folder, exist_ok=True)

    # Determine destination path
    file_name = os.path.basename(file_path)
    destination = os.path.join(current_folder, file_name)

    # Copy file
    shutil.copy(file_path, destination)

    return destination