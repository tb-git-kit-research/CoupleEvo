import os
import re
from typing import Tuple

from dotenv import load_dotenv


def _scenario_folder(instance_file: str) -> str:
    """
    Extract the scenario folder.
    Works for:
      - U_C_N25_Q8_T6.txt
    """
    # Example: U_C_N25_Q8_T6.txt --> Urban_C
    region = instance_file.split("_")[0]
    mode = instance_file.split("_")[1]
    region_name = "Rural" if region == "R" else "Urban"
    return f"{region_name}_{mode}"


def _instance_stem(instance_file: str) -> str:
    """
    Extract the instance stem without extension.
    Works for:
      - U_C_N25_Q8_T6.txt
    """
    return "N" + instance_file.split("N")[1].strip(".txt")

def load_branch_and_cut_stats(
    instance_file: str,
) -> Tuple[float, float, float]:
    """
    Returns:
        (objective_value, solution_time, mip_gap)
    """

    base_path = os.environ["BASE_PATH"]

    stem = _instance_stem(instance_file)
    scenario = _scenario_folder(instance_file)

    bc_path = os.path.join(
        base_path,
        "data",
        "inventor_routing_instances",
        "SKALNES",
        "benchmark_results",
        "Results_BC",
        scenario,
        f"D1_{stem}_P1_C1.txt"
    )

    if not os.path.isfile(bc_path):
        raise FileNotFoundError(f"BC solution not found: {bc_path}")

    z = None
    time = None
    gap = None

    z_re = re.compile(r"^z\s+([-\deE\.]+)\s*$")
    time_re = re.compile(r"^Solution_time\s+([-\deE\.]+)\s*$")
    gap_re = re.compile(r"^MIP_Gap\s+([-\deE\.]+)\s*$")

    with open(bc_path, "r") as f:
        for line in f:
            line = line.strip()

            if z is None:
                m = z_re.match(line)
                if m:
                    z = float(m.group(1))
                    continue

            if time is None:
                m = time_re.match(line)
                if m:
                    time = float(m.group(1))
                    continue

            if gap is None:
                m = gap_re.match(line)
                if m:
                    gap = float(m.group(1))
                    continue

            if z is not None and time is not None and gap is not None:
                break



    return z, time, gap



if __name__ == "__main__":
    load_dotenv()

    z, t, gap = load_branch_and_cut_stats(
        instance_file="U_C_N25_Q8_T6.txt",
    )

    print(f"BC objective = {z}, time = {t:.2f}s, {gap} gap")

