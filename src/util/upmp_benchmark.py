import os
import re
import json
import sys
from typing import List, Dict


def parse_experiment_string(exp_string: str) -> Dict:
    pattern = (
        r"exp_bay_(?P<bay>\d+)_wh_(?P<wh>\d+)_fill_(?P<fill>[0-9.]+)"
        r"_seed_(?P<seed_start>\d+)-(?P<seed_end>\d+)"
        r"_priority_(?P<priority>\d+)_robots_(?P<robots>\d+)"
        r"_ad_(?P<ad>\d+)\.json"
    )

    match = re.match(pattern, exp_string)
    if not match:
        raise ValueError(f"Invalid experiment string: {exp_string}")

    d = match.groupdict()
    d["seed_start"] = int(d["seed_start"])
    d["seed_end"] = int(d["seed_end"])
    d["fill"] = float(d["fill"])

    return d


def find_matching_files(experiments_dir: str, params: Dict) -> List[str]:
    matched_files = []

    fill_str = str(params["fill"])

    if not os.path.exists(experiments_dir):
        raise FileNotFoundError(f"Directory not found: {experiments_dir}")

    for root, dirs, files in os.walk(experiments_dir):
        # DEBUG: show traversal
        # print(f"[DEBUG] Entering: {root}")

        for f in files:
            if not f.endswith(".json"):
                continue

            filepath = os.path.join(root, f)

            # --- relaxed filtering (important for nested structures) ---
            if f"fill_lvl_{fill_str}" not in f:
                continue

            if f"directions_{params['ad']}" not in f:
                continue

            # robots filter sometimes missing in mixed → make optional
            if f"nr_r_{params['robots']}" not in f:
                # don't skip immediately → mixed files may not contain this
                pass

            # --- extract seed ---
            seed_match = re.search(r"seed_(\d+)", f)
            if not seed_match:
                continue

            seed = int(seed_match.group(1))

            if params["seed_start"] <= seed <= params["seed_end"]:
                matched_files.append(filepath)

    if not matched_files:
        print(f"[WARNING] No files matched in {experiments_dir}")

    return matched_files


def extract_metrics(filepath: str) -> Dict:
    with open(filepath, "r") as f:
        data = json.load(f)

    return {
        "file": os.path.basename(filepath),
        "astar_runtime": data.get("runtime"),
        "solver_runtime": data.get("solver_time"),
        "objective": max(data.get("robot")["traveled_times"]),
        "total_distance": data.get("total_distance"),
        "number_moves": data.get("f_final"),
        "bay": int(data.get("layout")[1]),
        "warehouse": int(data.get("layout")[-1]),
        "fill_level": data.get("fill_level"),
        "seed": data.get("seed"),
    }

def extract_metrics_mixed(filepath: str) -> Dict:
    print(filepath)
    with open(filepath, "r") as f:
        data = json.load(f)

    return {
        "file": os.path.basename(filepath),
        "objective": data.get("objective_value"),
        "seed": int(data.get("seed")),
        "bay": int(data.get("layout_file").split("Size_")[1][0]),
        "warehouse": int(data.get("layout_file").split("Layout_")[1][0]),
        "fill_level": data.get("fill_level"),
        "seed": data.get("seed"),
    }


def load_experiment_results(
    exp_string: str,
    exp_dir: str,
    root_folder: str = "experiments",
    mode: str = "mixed"
) -> List[Dict]:

    base_dir = os.getenv("BASE_PATH")

    experiments_dir = os.path.join(base_dir, root_folder, exp_dir)

    params = parse_experiment_string(exp_string)
    files = find_matching_files(experiments_dir, params)

    if mode == "sota":
        extractor = extract_metrics
    elif mode == "mixed":
        extractor = extract_metrics_mixed
    else:
        raise ValueError("mode must be 'sota' or 'mixed'")

    return [extractor(f) for f in files]


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    os.environ["BASE_PATH"] = r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling"
    exp_dir = "results_20251122_080703_multibay_reshuffle_astar_loaded_move_time_qwen25coder32b_ueTrue_uiFalse_uitypeNone"
    exp_string = (
            "exp_bay_5_wh_2_fill_0.8_seed_0-9_priority_5_robots_2_ad_1010.json"
        )

    try:
        results = load_experiment_results(exp_string, exp_dir)

        if not results:
            print("[WARNING] No matching files found.")
            sys.exit(0)

        results.sort(key=lambda x: x["seed"])

        print("\n[INFO] Results:")
        for r in results:
            print(
                f"seed={r['seed']} | "
                f"obj={r['objective']}"
            )

        objectives = [r["objective"] for r in results]

        print("\n[STATS]")
        print(f"Objective avg: {sum(objectives)/len(objectives):.2f}")

    except Exception as e:
        print(f"[ERROR] {e}")