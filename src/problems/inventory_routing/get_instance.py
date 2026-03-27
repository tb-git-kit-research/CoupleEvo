import os
from typing import Dict, List
import numpy as np

import re


def load_full_skalnes_instances(pattern: str) -> List[Dict]:
    """
    Loads SKALNES instances matching structured pattern.

    Example patterns:
        ?_?_N?_Q?_T?
        R_C_N?_Q?_T?
        R_C_N25_Q?_T?

    ? matches any sequence of digits/letters inside a segment.
    """

    base_path = os.environ["BASE_PATH"]

    instances_path = os.path.join(
        base_path,
        "data",
        "inventor_routing_instances",
        "SKALNES",
        "Instances",
    )

    if not os.path.isdir(instances_path):
        raise FileNotFoundError(instances_path)

    # ---------------------------------------------------------
    # Convert structured pattern to regex
    # ---------------------------------------------------------
    # Escape everything first
    regex_pattern = re.escape(pattern)

    # Replace escaped '?' with regex wildcard for segment
    regex_pattern = regex_pattern.replace(r"\?", r"[A-Za-z0-9]+")

    # Match full filename including .txt
    regex_pattern = f"^{regex_pattern}\\.txt$"

    compiled = re.compile(regex_pattern)

    # ---------------------------------------------------------
    # Match files
    # ---------------------------------------------------------
    files = sorted(
        f for f in os.listdir(instances_path)
        if compiled.match(f)
    )

    if not files:
        raise ValueError(f"No instances found for pattern '{pattern}'")

    instances = [
        _parse_full_skalnes_instance(os.path.join(instances_path, f))
        for f in files
    ]

    # ---------------------------------------------------------
    # Sort: Customers → Capacity → Periods
    # ---------------------------------------------------------
    instances.sort(
        key=lambda inst: (
            inst["meta"]["area_code"],
            inst["meta"]["dist_code"],
            inst["meta"]["nCustomers"],
            inst["vehicles"]["capacity"],
            inst["meta"]["nPeriods"],
        )
    )

    return instances


def _parse_full_skalnes_instance(filepath: str) -> Dict:
    import os
    import numpy as np

    # ---------------------------------------------------------
    # Extract filename metadata (area + distribution)
    # ---------------------------------------------------------
    filename = os.path.basename(filepath)
    name_no_ext = filename.replace(".txt", "")
    parts = name_no_ext.split("_")

    if len(parts) < 5:
        raise ValueError(f"Unexpected filename format: {filename}")

    area_code = parts[0]
    dist_code = parts[1]


    # ---------------------------------------------------------
    # Read file
    # ---------------------------------------------------------
    with open(filepath, "r") as f:
        lines = [line.rstrip() for line in f if line.strip()]

    # ---------------------------------------------------------
    # Global parameters
    # ---------------------------------------------------------
    nVehicles  = int(_find_value(lines, "nVehicles"))
    nDepots    = int(_find_value(lines, "nDepots"))
    nCustomers = int(_find_value(lines, "nCustomers"))
    nPeriods   = int(_find_value(lines, "nPeriods"))

    # ---------------------------------------------------------
    # Vehicle fleet
    # ---------------------------------------------------------
    vehicle_idx = _find_line_index(lines, "VEHICLE_TYPE")
    vt = lines[vehicle_idx + 1].split()

    vehicle_type = int(vt[0])
    fleet_size   = int(vt[1])
    capacity     = float(vt[2])

    vehicles = {
        "type": vehicle_type,
        "fleet_size": fleet_size,
        "capacity": capacity,
    }

    # ---------------------------------------------------------
    # Depot
    # ---------------------------------------------------------
    depot_idx = _find_line_index(lines, "DEPOT")
    depot_row = lines[depot_idx + 1].split()

    depot = {
        "id": int(depot_row[0]),
        "x": float(depot_row[1]),
        "y": float(depot_row[2]),
        "inventory_cost": float(depot_row[3]),
        "initial_inventory": float(depot_row[4]),
        "min_inventory": float(depot_row[5]),
        "max_inventory": float(depot_row[6]),
        "production": [
            float(v) for v in depot_row[7:7 + nPeriods]
        ],
    }

    # ---------------------------------------------------------
    # Customers
    # ---------------------------------------------------------
    customer_idx = _find_line_index(lines, "CUSTOMER")
    customers = []

    for i in range(nCustomers):
        row = lines[customer_idx + 1 + i].split()

        customers.append({
            "id": int(row[0]),
            "x": float(row[1]),
            "y": float(row[2]),
            "inventory_cost": float(row[3]),
            "initial_inventory": float(row[4]),
            "min_inventory": float(row[5]),
            "max_inventory": float(row[6]),
            "demand": [
                float(v) for v in row[7:7 + nPeriods]
            ],
        })

    # ---------------------------------------------------------
    # Cost matrix (robust to triangular input)
    # ---------------------------------------------------------
    cost_idx = _find_line_index(lines, "COST_MATRIX")
    matrix_size = 1 + nCustomers  # depot + customers

    cost_matrix = np.zeros((matrix_size, matrix_size), dtype=float)

    # Skip header row (node labels)
    start = cost_idx + 2

    for r in range(matrix_size):
        tokens = lines[start + r].split()
        if not tokens:
            continue

        row_node = int(tokens[0])
        values = tokens[1:]

        for col, val in enumerate(values):
            dist = float(val)
            cost_matrix[row_node, col] = dist
            cost_matrix[col, row_node] = dist

    # ---------------------------------------------------------
    # Final structured instance
    # ---------------------------------------------------------
    return {
        "meta": {
            "file": filename,
            "area_code": area_code,
            "dist_code": dist_code,
            "nVehicles": nVehicles,
            "nDepots": nDepots,
            "nCustomers": nCustomers,
            "nPeriods": nPeriods,
        },
        "vehicles": vehicles,
        "depot": depot,
        "customers": customers,
        "cost_matrix": cost_matrix,
    }


def _find_value(lines: List[str], key: str) -> str:
    for line in lines:
        if line.startswith(key):
            return line.split()[-1]
    raise ValueError(f"Key '{key}' not found")


def _find_line_index(lines: List[str], key: str) -> int:
    for i, line in enumerate(lines):
        if line.startswith(key):
            return i
    raise ValueError(f"Section '{key}' not found")


if __name__ == "__main__":

    os.environ["BASE_PATH"] = (
        r"C:\Users\dq0913\PycharmProjects\multibay-reshuffling"
    )

    instances = load_full_skalnes_instances("R_?_N?_Q?_T?")

    print(f"\nLoaded {len(instances)} instances\n")

    for inst in instances:
        #for entry in inst:
        #    print(entry, inst[entry])
        print(f"Instance: {inst['meta']['file']}")
        #print(f"  Customers : {inst['meta']['nCustomers']}")
        #print(f"  Periods   : {inst['meta']['nPeriods']}")
        #print(f"  Vehicles  : {inst['vehicles']['fleet_size']} "
        #      f"(cap={inst['vehicles']['capacity']})")
        #print(f"  Depot init inventory: {inst['depot']['initial_inventory']}")
        #print(f"  First customer demand vector length: "
        #      f"{len(inst['customers'][0]['demand'])}")
        #print(f"  Cost matrix shape: {inst['cost_matrix'].shape}")
        #print("-" * 60)
