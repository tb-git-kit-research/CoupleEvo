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

import os
import re
import json
import tempfile
import autopep8
import torch
import numpy as np
import pandas as pd

# Tree-sitter imports for code parsing and comment/docstring removal
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# Transformer and clustering imports
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import DBSCAN


# -------------------- Data Extraction Functions --------------------

def extract_population_from_json_file(json_file):
    """
    Extracts the population number from the json_file name.
    """
    match = re.search(r"program_pop_(\d+)", json_file)
    return int(match.group(1)) if match else None


def extract_ue_flag_from_json_file(run_folder):
    """
    Checks if 'ueTrue' is in run_folder name.
    """
    return "ueTrue" in run_folder


def extract_ui_flag_from_json_file(run_folder):
    """
    Checks if 'uiTrue' is in run_folder name.
    """
    return "uiTrue" in run_folder


def extract_ui_type_from_json_file(run_folder):
    """
    Extracts the ui_type from the run_folder name.
    """
    try:
        return run_folder.split("uitype")[1]
    except Exception:
        return "None"


def extract_prompt_strategy_from_json_file(json_file):
    """
    Extracts the prompt strategy from the json_file name.
    """
    try:
        return json_file.split("_")[4]
    except Exception:
        return "None"


def extract_offspring_data_from_json_data(model_folder, json_file, run_folder, json_data):
    """
    Extracts detailed data from the 'p' field and other relevant fields.
    Returns a list of flattened dictionaries.
    """
    data_records = []
    data_records.append({
        "model_folder": model_folder,
        "model": json_data.get("full_request", {}).get("model"),
        "run_folder": run_folder,
        "json_file": json_file,
        "population": extract_population_from_json_file(json_file),
        "ue_flag": extract_ue_flag_from_json_file(run_folder),
        "ui_flag": extract_ui_flag_from_json_file(run_folder),
        "ui_type": extract_ui_type_from_json_file(run_folder),
        "algorithm": json_data.get("offspring", {}).get("algorithm", [None])[0],
        "code": " ".join(json_data.get("offspring", {}).get("code", [])),
        "objective": json_data.get("offspring", {}).get("objective"),
        "evaluation_time": json_data.get("offspring", {}).get("evaluation_time"),
        "offspring_id": json_data.get("offspring", {}).get("offspring_id"),
        "prompt_strategy": extract_prompt_strategy_from_json_file(json_file),
    })
    return data_records


def read_and_process_runs(base_directory, filter_model_folders):
    """
    Reads JSON data from all 'all_programs' folders in the given directory,
    extracts detailed data, and returns a Pandas DataFrame.
    Only processes model folders specified in `filter_model_folders`.
    """
    all_data_records = []
    for model_folder in os.listdir(base_directory):
        if model_folder not in filter_model_folders:
            continue  # Skip unwanted model folders
        model_path = os.path.join(base_directory, model_folder)
        if os.path.isdir(model_path):
            print("Processing model folder:", model_folder)
            for experiment_folder in os.listdir(model_path):
                experiment_path = os.path.join(model_path, experiment_folder)
                if os.path.isdir(experiment_path):
                    all_programs_folder = os.path.join(experiment_path, "all_programs")
                    if os.path.exists(all_programs_folder) and os.path.isdir(all_programs_folder):
                        json_files = [f for f in os.listdir(all_programs_folder) if f.endswith(".json")]
                        for json_file in json_files:
                            file_path = os.path.join(all_programs_folder, json_file)
                            try:
                                with open(file_path, 'r') as file:
                                    json_data = json.load(file)
                                    all_data_records.extend(
                                        extract_offspring_data_from_json_data(
                                            model_folder, json_file, experiment_folder, json_data
                                        )
                                    )
                            except Exception as e:
                                # Optionally log error: print(f"Error processing file {file_path}: {e}")
                                pass
    df = pd.DataFrame(all_data_records) if all_data_records else pd.DataFrame()
    return df


def assign_experiment(row):
    """
    Assigns an experiment type based on ue_flag, ui_flag, and ui_type.
    """
    if row["ue_flag"] and row["ui_flag"]:
        return f"CEoH + {row['ui_type']}"
    elif row["ue_flag"]:
        return "CEoH"
    elif row["ui_flag"]:
        return f"EoH + {row['ui_type']}"
    elif not row["ue_flag"] and not row["ui_flag"]:
        return "EoH"
    else:
        return None


# -------------------- Code Formatting and Cleaning --------------------

# Load Python parser for tree-sitter
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def format_python_code(code_str):
    """Formats Python code using autopep8."""
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.py') as temp_file:
        temp_file.write(code_str)
        temp_filename = temp_file.name
    autopep8_args = ['--in-place', '--aggressive', '--aggressive', temp_filename]
    autopep8.fix_file(temp_filename, options=autopep8.parse_args(autopep8_args))
    with open(temp_filename, 'r') as temp_file:
        formatted_code = temp_file.read()
    return formatted_code


def remove_comments_and_docstrings(code):
    """Removes comments and docstrings from Python code using tree-sitter."""
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    lines = code.split('\n')
    to_remove = []

    def is_comment(node):
        return node.type == 'comment'

    def is_docstring(node):
        return (node.type == 'string' and node.parent and node.parent.type in ['expression_statement', 'module'])

    def traverse(node):
        if is_comment(node) or is_docstring(node):
            to_remove.append((node.start_point, node.end_point))
        for child in node.children:
            traverse(child)

    traverse(root_node)

    for start, end in reversed(to_remove):
        start_row, start_col = start
        end_row, end_col = end
        if start_row == end_row:
            lines[start_row] = lines[start_row][:start_col] + lines[start_row][end_col:]
        else:
            lines[start_row] = lines[start_row][:start_col]
            for row in range(start_row + 1, end_row):
                lines[row] = ''
            lines[end_row] = lines[end_row][end_col:]
    cleaned_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(cleaned_lines)


def preprocess_code_for_formatting(code):
    """Formats code and removes comments/docstrings for embedding."""
    # Handle Unicode issues by encoding to ascii and ignoring errors.
    code = code.encode('ascii', 'ignore').decode('ascii')
    formatted_code = format_python_code(code)
    cleaned_code = remove_comments_and_docstrings(formatted_code)
    return cleaned_code


# -------------------- Embedding and Diversity Analysis --------------------

# Model setup (CodeBERT)
checkpoint = "microsoft/codebert-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)


def preprocess_code_for_embedding(code):
    """Simple preprocessing for code before embedding."""
    return code.strip()


def get_embedding(code):
    """Generates an embedding for a code snippet."""
    preprocessed_code = preprocess_code_for_embedding(code)
    inputs = tokenizer.encode_plus(
        preprocessed_code, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding.flatten()


def get_embeddings_batch(codes):
    """
    Batch embedding: preprocesses a list of code snippets,
    tokenizes, and computes embeddings.
    """
    preprocessed = [preprocess_code_for_embedding(code) for code in codes]

    inputs = tokenizer(preprocessed, return_tensors="pt", truncation=True,
                       max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


def calculate_cdi(embeddings):
    """
    Computes the Community Diversity Index (CDI) using the Minimum Spanning Tree.
    """
    embeddings_matrix = np.vstack(embeddings)
    distance_matrix = np.linalg.norm(embeddings_matrix[:, None] - embeddings_matrix, axis=2)
    mst = minimum_spanning_tree(distance_matrix).toarray()
    mst_distances = mst[mst != 0]
    total_distance = np.sum(mst_distances)
    proportions = mst_distances / total_distance
    diversity_index = -np.sum(proportions * np.log(proportions))
    return diversity_index


# -------------------- Diversity Computation Functions --------------------



def compute_embedding_diversity_cumulative(df, threshold=0.8, group_by=None, output_dir_cdi="diversity_analysis"):
    """
    Computes diversity metrics cumulatively for each population level in each group,
    and saves separate CSV files per population.
    """
    os.makedirs(output_dir_cdi, exist_ok=True)
    filename_regex = re.compile(r'[^a-zA-Z0-9._-]')
    grouped = df.groupby(group_by)
    all_results = []

    for group_name, group_df in grouped:
        print(f"Processing group: {group_name}")

        # check if group was already processed
        if os.path.exists(os.path.join(output_dir_cdi, group_df["run_folder"].values[0])):
            print(f"Skipping group {group_name} (already processed)")
            continue
        else:
            os.makedirs(os.path.join(output_dir_cdi, group_df["run_folder"].values[0]), exist_ok=True)

        codes = group_df['code'].tolist()
        embeddings_group = get_embeddings_batch(codes)  # Batch processing for efficiency
        # save embedings to file for further analysis
        run_folder = group_df["run_folder"].values[0]
        embeddings_file = os.path.join(output_dir_cdi, run_folder, f"embeddings.npy")
        np.save(embeddings_file, embeddings_group)
        print("saved embeddings to", embeddings_file)

        populations = group_df["population"].values

        for population in range(0, group_df["population"].max() + 1):
            sanitized_group_name = filename_regex.sub('_', str(group_name))

            group_dir = os.path.join(output_dir_cdi, run_folder)
            os.makedirs(group_dir, exist_ok=True)
            cdi_file = os.path.join(group_dir, f"diversity_{sanitized_group_name}_pop_{population}.csv")
            if os.path.exists(cdi_file):
                print(f"Skipping group {group_name} for population {population} (file exists)")
                continue
            mask = populations <= population
            embeddings = embeddings_group[mask]
            if len(embeddings) < 2:
                print(f"Skipping group {group_name} for population {population} (not enough snippets)")
                continue
            cdi = calculate_cdi(embeddings)
            group_df_filter = group_df[mask]
            result = {
                "model_folder": group_df_filter["model_folder"].values[0],
                "group": group_name,
                "population_limit": population,
                "experiment": group_df_filter["experiment"].values[0],
                "model": group_df_filter["model"].values[0],
                "run_folder": run_folder,
                "prompt_strategy": group_df_filter["prompt_strategy"].values[0],
                "max_objective": group_df_filter["objective"].max(),
                "min_objective": group_df_filter["objective"].min(),
                "median_objective": group_df_filter["objective"].median(),
                "mean_objective": group_df_filter["objective"].mean(),
                "std_objective": group_df_filter["objective"].std(),
                "total_diversity_cdi": cdi,
            }
            result_df = pd.DataFrame([result])
            result_df.to_csv(cdi_file, index=False)
            print(f"Processed group: {group_name} for population {population}, saved to {cdi_file}")
            all_results.append(result)

    print(f"All results saved in '{output_dir_cdi}' directory.")
    return pd.DataFrame(all_results)


def run_cdi(base_dir, output_dir, filter_model_folders=["diversity_folder"]):
    """
    Processes runs from the base directory, cleans and formats the code,
    computes cumulative diversity metrics, and saves results in the output directory.

    Parameters:
        base_dir (str): Path to the base directory containing JSON run folders.
        output_dir (str): Path to the output directory where results will be saved.
        filter_model_folders (list): List of model folder names to process.

    Returns:
        pd.DataFrame: DataFrame containing overall diversity results.
    """
    # Process the runs and extract data into a DataFrame
    results_df = read_and_process_runs(base_dir, filter_model_folders)

    #filter first 10 rows of each population
    #filter_number = 2
    #results_df = results_df.groupby(['model_folder', 'run_folder', 'population']).head(filter_number).reset_index()
    #print(f"Filtered first {filter_number} rows of each population")
    #print(results_df)

    # Assign experiment type based on flags
    if not results_df.empty:
        results_df["experiment"] = results_df.apply(assign_experiment, axis=1)
    else:
        print("No results found. Exiting.")
        return None

    # Clean and format code for embedding
    results_df["code"] = results_df["code"].apply(preprocess_code_for_formatting)

    # Sort the DataFrame by specified columns
    results_df = results_df.sort_values(by=['model', 'run_folder', 'population', 'experiment', 'prompt_strategy'],
                                        ascending=False)

    # (Optional) Filter dataset for populations less than a threshold (e.g., 20)
    results_df_filter = results_df[results_df['population'] < 20]

    # Define output directory for overall diversity analysis and create it if necessary
    overall_diversity_dir = os.path.join(output_dir, "diversity_analysis_overall")
    os.makedirs(overall_diversity_dir, exist_ok=True)

    # Set ordering for prompt strategies (if applicable)
    prompt_strategy_order = ['i1', 'e1', 'e2', 'p1', 'm1', 'm2']
    results_df["prompt_strategy"] = pd.Categorical(results_df["prompt_strategy"],
                                                   categories=prompt_strategy_order, ordered=True)

    # Compute overall diversity using cumulative method
    diversity_df_overall = compute_embedding_diversity_cumulative(
        results_df_filter, threshold=0.95,
        group_by=['model', 'run_folder', "experiment"],
        output_dir_cdi=overall_diversity_dir
    )

    print("Processing complete.")
    return diversity_df_overall


if __name__ == "__main__":
    # Base directories (adjust paths as needed)
    base_dir = r"C:\Users\smthboem\PycharmProjects\multibay-reshuffling\eoh_results"
    output_dir = r"C:\Users\smthboem\PycharmProjects\multibay-reshuffling\eoh_analysis"
    filter_model_folders = ["qwen25coder32b_ueTrue_uiTrue"]

    # Run the CDI computation workflow
    diversity_df_overall = run_cdi(base_dir, output_dir, filter_model_folders)
    if diversity_df_overall is not None:
        print(diversity_df_overall)
