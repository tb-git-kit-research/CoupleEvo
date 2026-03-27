import os
from typing import Optional
import pandas as pd
from dotenv import load_dotenv


def _parse_instance_name(instance_file: str) -> str:
    """
    U_C_N25_Q8_T12.txt -> U_C_N25_Q8_T12
    """
    return os.path.splitext(os.path.basename(instance_file))[0]


def _to_float(x) -> Optional[float]:
    """Safe float conversion (handles NaN)."""
    if pd.isna(x) or x == "":
        return None
    return float(x)


def load_hgs_stats(instance_file: str):
    """
    Returns:
        (objective_value, runtime)
    Objective = Best (HGS solution)
    Runtime   = tHGS
    """

    base_path = os.environ["BASE_PATH"]

    csv_path = os.path.join(
        base_path,
        "data",
        "inventor_routing_instances",
        "SKALNES",
        "benchmark_results",
        "HGS-IRP.csv",
    )

    df = pd.read_csv(csv_path)

    instance = _parse_instance_name(instance_file)

    row = df[df["Instance"] == instance]

    if row.empty:
        raise FileNotFoundError(f"No HGS result found for {instance}")

    objective = _to_float(row.iloc[0]["sHGS"])
    runtime = _to_float(row.iloc[0]["tHGS"])

    return objective, runtime

def load_all_hgs_stats(instances):
    objectives = []
    for instance in instances:
        objective, runtime = load_hgs_stats(instance["meta"]["file"])
        objectives.append(objective)
    return objectives

if __name__ == "__main__":
    load_dotenv()

    z, t = load_hgs_stats("R_C_N50_Q38_T6.txt")
    print(f"HGS objective = {z}, time = {t}s")
