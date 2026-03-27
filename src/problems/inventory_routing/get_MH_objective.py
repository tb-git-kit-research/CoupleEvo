import os
from typing import Tuple
import pandas as pd
from dotenv import load_dotenv


def _parse_instance(instance_file: str):
    """
    U_C_N25_Q8_T12.txt -> ('Urban', 'C', 25, 8, 12)
    """
    region, mode, rest = instance_file.split("_", 2)
    region_name = "Urban" if region == "U" else "Rural"

    parts = rest.replace(".txt", "").split("_")
    N = int(parts[0][1:])
    Q = int(parts[1][1:])
    T = int(parts[2][1:])

    return region_name, mode, N, Q, T


def load_matheuristic_stats(instance_file: str) -> Tuple[float, float, float]:
    """
    Returns:
        (objective_value, run_time, mip_gap)
    """

    base_path = os.environ["BASE_PATH"]

    csv_path = os.path.join(
        base_path,
        "data",
        "inventor_routing_instances",
        "SKALNES",
        "benchmark_results",
        "Results_Matheuristic",
        "Matheuristic_IRP.csv",
    )

    df = pd.read_csv(csv_path, delimiter=";", decimal=",")

    region, mode, N, Q, T = _parse_instance(instance_file)

    # ---- adapt column names here if needed ----
    row = df[
        (df["Region"] == region)
        & (df["Mode"] == mode)
        & (df["N"] == N)
        & (df["Q"] == Q)
        & (df["T"] == T)
    ]

    if row.empty:
        raise FileNotFoundError(f"No matheuristic result for {instance_file}")

    def _to_float_eu(x):
        if pd.isna(x):
            return None
        return float(str(x).replace(".", "").replace(",", "."))

    obj = _to_float_eu(row.iloc[0]["UB"])
    time = _to_float_eu(row.iloc[0]["Seconds"])

    return obj, time, None


if __name__ == "__main__":
    load_dotenv()

    z, t, gap = load_matheuristic_stats("U_C_N25_Q8_T12.txt")
    print(f"MH objective = {z}, time = {t}s, gap = {gap}")
