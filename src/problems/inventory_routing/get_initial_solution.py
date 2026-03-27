import os
import re
import numpy as np
from typing import Dict, Any

import numpy as np


def _scenario_folder(eoh_experiment_file: str) -> str:
    # Example: R_RC_N10 --> Rural_RC
    region = eoh_experiment_file.split("_")[0]
    mode = eoh_experiment_file.split("_")[1]

    region_name = "Rural" if region == "R" else "Urban"
    return f"{region_name}_{mode}"


def load_initial_construction_solution_with_routes(instance_file: str) -> Dict[str, Any]:
    base_path = os.environ["BASE_PATH"]

    name = os.path.splitext(os.path.basename(instance_file))[0]
    parts = name.split("_")

    # Example: R_C_N25_Q8_T6
    # We extract the numeric parts explicitly
    N = int(next(p[1:] for p in parts if p.startswith("N")))
    T = int(next(p[1:] for p in parts if p.startswith("T")))
    Q = next(p[1:] for p in parts if p.startswith("Q"))

    scenario = _scenario_folder(instance_file)
    sol_file = f"D1_N{N}_Q{Q}_T{T}_P1_C1.txt_user.result.txt"

    sol_path = os.path.join(
        base_path,
        "data",
        "inventor_routing_instances",
        "SKALNES",
        "benchmark_results",
        "Results_Constructive",
        scenario,
        sol_file
    )

    if not os.path.isfile(sol_path):
        raise FileNotFoundError(f"Initial solution not found: {sol_path}")

    # We'll build these
    shipments = None                  # shape (T, V, N) once V known
    delivered_by_tv = {}              # temp: (t, v) -> np.array length N
    visit_order = {}                  # t -> {v -> [cust_idx...]}

    # State
    in_delivered = False
    current_delivered_t = None

    current_route_t = None
    current_route_v = None
    in_routes = False

    # Helpers
    delivered_period_re = re.compile(r"^period:\s*(\d+)\s*$")         # delivered section (1-based)
    delivered_vehicle_re = re.compile(r"^vehicle:\s*(\d+)\s+(.*)$")   # "vehicle: k <numbers...>"

    route_period_re = re.compile(r"^period\s+(\d+):\s*$")             # route section (0-based)
    route_vehicle_re = re.compile(r"^vehicle\s+(\d+):\s*$")           # "vehicle k:"
    route_path_re = re.compile(r"^\s*0(-->\d+)+-->0\s*$")             # "0-->2-->0" etc.

    with open(sol_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Enter delivered quantities section
            if line.startswith("Optimal Delivered quantities"):
                in_delivered = True
                in_routes = False
                continue

            # Enter routes section (it appears after "Optimal Z values" in this file)
            # Easiest trigger: first "period X:" that is NOT "period:" form.
            if route_period_re.match(line):
                in_routes = True
                in_delivered = False

            # ---- Delivered quantities parsing ----
            if in_delivered:
                m = delivered_period_re.match(line)
                if m:
                    # delivered periods are 1-based in the file
                    current_delivered_t = int(m.group(1)) - 1
                    continue

                m = delivered_vehicle_re.match(line)
                if m and current_delivered_t is not None:
                    v = int(m.group(1)) - 1  # vehicles are 1-based in file
                    nums = m.group(2).split()

                    # Some files include depot column; others may not.
                    # In your N10 file, the route nodes are 1..10 customers, depot is 0,
                    # and delivered vectors appear to be length N (customers only). :contentReference[oaicite:2]{index=2}
                    values = np.array([float(x) for x in nums], dtype=float)

                    # Normalize to length N by:
                    # - if length is N+1, assume leading depot column and drop it
                    # - if length is N, keep as-is
                    # - otherwise, raise to catch format drift early
                    if len(values) == N + 1:
                        values = values[1:]
                    elif len(values) != N:
                        raise ValueError(f"Unexpected delivered vector length {len(values)} (expected {N} or {N+1}) in {sol_path}")

                    delivered_by_tv[(current_delivered_t, v)] = values
                    continue

            # ---- Route section parsing ----
            if in_routes:
                m = route_period_re.match(line)
                if m:
                    current_route_t = int(m.group(1))  # already 0-based
                    visit_order.setdefault(current_route_t, {})
                    current_route_v = None
                    continue

                m = route_vehicle_re.match(line)
                if m and current_route_t is not None:
                    current_route_v = int(m.group(1)) - 1
                    visit_order[current_route_t].setdefault(current_route_v, [])
                    continue

                # Path line can be on next line after "vehicle k:"
                # Example: "0-->2-->0" :contentReference[oaicite:3]{index=3}
                if current_route_t is not None and current_route_v is not None and "-->" in line and line.startswith("0"):
                    nodes = [int(x) for x in line.split("-->")]
                    # nodes like [0, 2, 0]. Customers are 1..N, depot is 0.
                    cust_nodes = [n for n in nodes if n != 0]
                    cust_idx = [n - 1 for n in cust_nodes]  # convert 1..N -> 0..N-1
                    visit_order[current_route_t][current_route_v] = cust_idx
                    continue

    # Infer V from delivered_by_tv or visit_order
    max_v = -1
    for (_, v) in delivered_by_tv.keys():
        max_v = max(max_v, v)
    for t in visit_order:
        for v in visit_order[t].keys():
            max_v = max(max_v, v)
    V = max_v + 1 if max_v >= 0 else 0

    shipments = np.zeros((T, V, N), dtype=float)
    routes = {t: {v: [] for v in range(V)} for t in range(T)}

    # Fill shipments and routes (merge delivered quantities + visit order)
    for t in range(T):
        for v in range(V):
            qvec = delivered_by_tv.get((t, v), None)
            if qvec is None:
                continue

            # shipments
            shipments[t, v, :] = qvec

            # route order: if missing, fall back to customers with q>0 in ascending index
            order = visit_order.get(t, {}).get(v, None)
            if order is None:
                order = list(np.where(qvec > 1e-9)[0])

            # build route representation [(i, q_i), ...] in that order
            routes[t][v] = [(i, float(qvec[i])) for i in order if qvec[i] > 1e-9]

    return {"shipments": shipments, "routes": routes}
