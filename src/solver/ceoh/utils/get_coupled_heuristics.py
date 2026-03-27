import os
import json
import glob
import re
import dotenv

from solver.ceoh.utils.getParas import Paras


def get_code_strings(paras):
    """
    Returns the best heuristic code per subproblem.

    If no heuristic is found for a subproblem, the value is None.
    """

    if not paras.coupled_problems:
        raise ValueError("paras.coupled_problems is not set")

    current_folder = os.environ.get("CURRENT_EXPERIMENT")
    if current_folder is None:
        raise EnvironmentError("CURRENT_EXPERIMENT is not set")

    pops_best_path = os.path.join(current_folder, "pops_best")

    # If no pops_best folder exists, return None for all subproblems
    if not os.path.isdir(pops_best_path):
        return {
            subproblem: None
            for subproblem in (
                paras.coupled_problems
                if isinstance(paras.coupled_problems, (list, tuple))
                else [paras.coupled_problems]
            )
        }

    # Normalize subproblem list
    if isinstance(paras.coupled_problems, str):
        subproblems = [
            s.strip() for s in paras.coupled_problems.split(",") if s.strip()
        ]
    else:
        subproblems = list(paras.coupled_problems)

    json_files = glob.glob(os.path.join(pops_best_path, "*.json"))

    # Initialize result with None for all subproblems
    best_per_subproblem = {subproblem: None for subproblem in subproblems}

    if not json_files:
        return best_per_subproblem

    for subproblem in subproblems:
        best_obj = float("inf")
        best_code = None

        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                entry = json.load(f)

            offspring_id = entry.get("offspring_id", "")
            if subproblem not in offspring_id:
                continue

            obj = entry.get("objective")
            code = entry.get("code")

            if obj is None or code is None:
                continue

            if obj < best_obj:
                best_obj = obj
                best_code = code

        # best_code may remain None --> this is now valid and intentional
        best_per_subproblem[subproblem] = best_code

    return best_per_subproblem

import re

def extract_heuristics(response: str):
    heuristics = []

    blocks = re.split(r"\n\s*---+\s*\n", response.strip())

    for block in blocks:
        if not block.strip():
            continue

        idx_match = re.search(r"SUBPROBLEM\s+(\d+)", block)
        if not idx_match:
            raise ValueError(f"Missing SUBPROBLEM index in block:\n{block}")
        subproblem_index = int(idx_match.group(1))

        thoughts_match = re.search(r"THOUGHTS:\s*(.*)", block)
        if not thoughts_match:
            raise ValueError(f"Missing THOUGHTS in block:\n{block}")
        thoughts = thoughts_match.group(1).strip()

        code_match = re.search(r"```(?:python)?\s*(.*?)```", block, re.S | re.I)
        if not code_match:
            raise ValueError(f"Missing code block in block:\n{block}")
        code = code_match.group(1).strip()

        heuristics.append({
            "subproblem_index": subproblem_index,
            "thoughts": thoughts,
            "code": code
        })

    return heuristics


if __name__ == "__main__":

    print("Start LNS for CYCLE MOVE + ROUTE")

    # subprocess.run(["pip", "install", "."], stdout=subprocess.DEVNULL)
    coupled_problems = ["lns_multi_robot_upmp_move_search", "lns_multi_robot_upmp_vrp"]
    dotenv.load_dotenv()

    os.environ["CURRENT_EXPERIMENT"] = r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling\eoh_results\results_20251216_125318_lns_multi_robot_upmp_coupled_qwen25coder32b_ueFalse_uiFalse_uitypeNone"

    paras = Paras()
    # add eoh_experiment_file to paras
    paras.set_paras(
        target_subproblem="lns_multi_robot_upmp_vrp",
        coupled_problems=coupled_problems,
    )
    code_list = get_code_strings(paras)
    for c in code_list:
        print("##### #####")
        print(c)
        print(code_list[c])



