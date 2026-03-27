import os
from typing import Tuple
from dotenv import load_dotenv


def _parse_instance(instance_file: str) -> Tuple[str, str]:
    """
    Example:
        U_R_N10_Q18_T9.txt -> ("Urban", "R")
        R_C_N25_Q8_T12.txt -> ("Rural", "C")
    """
    region_code, mode, *_ = instance_file.split("_")
    region = "Urban" if region_code == "U" else "Rural"
    return region, mode


def _construct_result_filename(instance_file: str) -> str:
    """
    Converts:
        U_R_N10_Q18_T9.txt
    into:
        D1_N10_Q18_T9_P1_C1.txt_user.result.txt
    """
    core = instance_file.replace(".txt", "")
    _, _, rest = core.split("_", 2)
    return f"D1_{rest}_P1_C1.txt_user.result.txt"


def load_constructive_objective(instance_file: str) -> float:
    """
    Returns:
        objective_value
    """

    base_path = os.environ["BASE_PATH"]

    region, mode = _parse_instance(instance_file)
    subfolder = f"{region}_{mode}"

    result_filename = _construct_result_filename(instance_file)

    result_path = os.path.join(
        base_path,
        "data",
        "inventor_routing_instances",
        "SKALNES",
        "benchmark_results",
        "Results_Constructive",
        subfolder,
        result_filename,
    )

    if not os.path.isfile(result_path):
        raise FileNotFoundError(result_path)

    with open(result_path, "r") as f:
        for line in f:
            if line.startswith("Solution objective:"):
                return float(line.split(":")[1].strip())

    raise ValueError(f"Objective not found in {result_filename}")


if __name__ == "__main__":
    load_dotenv()

    z = load_constructive_objective("U_R_N10_Q18_T9.txt")
    print(f"Constructive objective = {z}")
