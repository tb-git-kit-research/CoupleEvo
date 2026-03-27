# Copyright (c) 2025
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

##############################################################################
#                               UTILITY FUNCTIONS
##############################################################################
import json
import os
import re

import pandas as pd


def extract_population_from_json_file(json_file: str):
    """
    Extracts the population number from the json_file name
    that matches the pattern 'program_pop_<number>'.
    e.g., 'program_pop_123.json' -> 123
    """
    match = re.search(r"program_pop_(\d+)", json_file)
    return int(match.group(1)) if match else None


def extract_ue_flag_from_json_file(run_folder: str) -> bool:
    """
    Extracts the ue_flag from the folder name.
    Returns True if 'ueTrue' is in the folder name, else False.
    """
    return "ueTrue" in run_folder


def extract_ui_flag_from_json_file(run_folder: str) -> bool:
    """
    Extracts the ui_flag from the folder name.
    Returns True if 'uiTrue' is in the folder name, else False.
    """
    return "uiTrue" in run_folder


def extract_ui_type_from_json_file(run_folder: str) -> str:
    """
    Extracts the ui_type from the folder name (assuming something like 'uitypeXYZ').
    If not found, returns 'None'.
    """
    try:
        return run_folder.split("uitype")[1]
    except Exception:
        return "None"


def extract_prompt_strategy_from_json_file(json_file: str) -> str:
    """
    Extracts the prompt strategy from the json_file name.
    For example, if json_file is 'abc_def_ghi_strategy_xyz.json',
    it tries to return the 5th element split by underscore.

    Adjust the split index for your naming convention if needed.
    """
    try:
        return json_file.split("_")[4]
    except Exception:
        return "None"


def assign_experiment(row: dict) -> str:
    """
    Assigns an experiment type based on ue_flag, ui_flag, and ui_type.
    """
    if row["ue_flag"] and row["ui_flag"]:
        return f"CEoH + {row['ui_type']}"
    elif row["ue_flag"]:
        return "CEoH"
    elif row["ui_flag"]:
        return f"EoH + {row['ui_type']}"
    else:
        return "EoH"

def extract_sample_from_json_file(json_file):
    if "sample" in json_file:
        return json_file.split("sample")[1].split("_")[1].split(".")[0]
    else:
        return "None"


def extract_offspring_data_from_json_data(
        model_folder: str,
        json_file: str,
        run_folder: str,
        json_data: dict
):
    """
    Extracts detailed data from the 'offspring' field in the JSON data.
    Returns a list of flattened dictionaries (one for each row).
    """
    data_records = []

    # Prepare a temp dict to facilitate the experiment assignment
    temp_dict = {
        "ue_flag": extract_ue_flag_from_json_file(run_folder),
        "ui_flag": extract_ui_flag_from_json_file(run_folder),
        "ui_type": extract_ui_type_from_json_file(run_folder),
    }

    # Build the final record
    offspring_dict = json_data.get("offspring", {})
    full_request = json_data.get("full_request", {})

    data_records.append({
        "model": full_request.get("model"),
        "model_folder": model_folder,
        "run_folder": run_folder,
        "json_file": json_file,
        "population": extract_population_from_json_file(json_file),
        "ue_flag": temp_dict["ue_flag"],
        "ui_flag": temp_dict["ui_flag"],
        "ui_type": temp_dict["ui_type"],
        "algorithm": (offspring_dict.get("algorithm", [None])[0]
                      if isinstance(offspring_dict.get("algorithm"), list) else None),
        "code": (offspring_dict.get("code", [None])[0]
                 if isinstance(offspring_dict.get("code"), list) else None),
        "objective": offspring_dict.get("objective"),
        "evaluation_time": offspring_dict.get("evaluation_time"),
        "offspring_id": offspring_dict.get("offspring_id"),
        "experiment": assign_experiment(temp_dict),
        "prompt_strategy": extract_prompt_strategy_from_json_file(json_file),
        "sample" : extract_sample_from_json_file(json_file),
        "parents": json_data.get("p"),
    })

    return data_records


def read_and_process_runs_only_experiment(
    base_directory: str,
    experiment_folder: str,
    filter_model_folders: list = None,
    program_folder: str = "all_programs",
) -> pd.DataFrame:
    """
    Reads JSON data from a *specific* experiment folder across all model folders
    in the base_directory (unless a subset of model folders is provided via
    filter_model_folders).

    Expected directory structure:
        base_directory/
          <model_folder>/
            experiment_folder/
              all_programs/
                *.json

    :param base_directory: The root directory containing multiple <model_folder>/ subdirectories.
    :param experiment_folder: The name of the experiment folder (e.g., "results_YYYYMMDD_...").
    :param filter_model_folders: (Optional) A list of specific model folder names to process.
                                 If None, all model folders will be processed.
    :return: A pandas DataFrame with all extracted rows.
    """
    all_data_records = []
    print(f"Reading data from {base_directory}...")

    # Iterate through the base directory structure
    for model_folder in os.listdir(base_directory):
        model_path = os.path.join(base_directory, model_folder)

        # Skip if it's not a folder
        if not os.path.isdir(model_path):
            print(f"Skipping non-folder: {model_folder}")
            continue

        # If a filter list is provided, skip folders not in that list
        if filter_model_folders is not None and model_folder not in filter_model_folders:
            print(f"Skipping {model_folder} as it's not in the filter list.")
            continue

        # Construct the specific experiment path
        exp_path = os.path.join(model_path, experiment_folder)
        if not os.path.isdir(exp_path):
            print(f"Skipping {model_folder} as {experiment_folder} folder does not exist.")
            # If this model folder does not have the experiment_folder, skip
            continue

        # Now look for 'all_programs' under this experiment folder
        all_programs_folder = os.path.join(exp_path, program_folder)
        if os.path.exists(all_programs_folder) and os.path.isdir(all_programs_folder):
            json_files = [f for f in os.listdir(all_programs_folder) if f.endswith(".json")]
            for json_file in json_files:
                file_path = os.path.join(all_programs_folder, json_file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        json_data = json.load(file)
                        # Append all extracted rows from this JSON
                        all_data_records.extend(
                            extract_offspring_data_from_json_data(
                                model_folder=model_folder,
                                json_file=json_file,
                                run_folder=experiment_folder,
                                json_data=json_data
                            )
                        )
                except Exception as e:
                    # Optionally log or print the error
                    pass

    # Create a DataFrame from all collected records
    df = pd.DataFrame(all_data_records) if all_data_records else pd.DataFrame()
    print(f"Data extraction complete. Total records: {len(df)}")
    return df