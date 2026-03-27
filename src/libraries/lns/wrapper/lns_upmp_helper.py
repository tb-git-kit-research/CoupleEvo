import hashlib
import os
import pickle

def get_heuristic_id(code: str) -> str:
    """Creates a unique ID for the heuristic code."""
    h = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return h[:16]   # short but unique enough

def get_initial_solution_path(heuristic_id: str, eoh_file: str, seed: int) -> str:
    base = os.path.join(os.getenv("BASE_PATH"),"data", "mr_initial_solutions", heuristic_id, eoh_file)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{seed}.pkl")


def save_initial_solution(path: str, solution: list):
    with open(path, "wb") as f:
        pickle.dump(solution, f)


def load_initial_solution(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
