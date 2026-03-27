import time
from typing import Optional

from numpy.random import Generator

from alns.State import State


class MaxIterationsAndMaxRuntime:
    """
    Criterion that stops after a maximum number of iterations OR maximum of runtime.
    """

    def __init__(self, max_iterations: int, max_runtime: float):
        if max_iterations < 0:
            raise ValueError("max_iterations < 0 not understood.")

        self._max_iterations = max_iterations
        self._current_iteration = 0

        if max_runtime < 0:
            raise ValueError("max_runtime < 0 not understood.")

        self._max_runtime = max_runtime
        self._start_runtime: Optional[float] = None


    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def max_runtime(self) -> float:
        return self._max_runtime

    def __call__(self, rng: Generator, best: State, current: State) -> bool:
        self._current_iteration += 1

        if self._start_runtime is None:
            self._start_runtime = time.perf_counter()

        iter_stop = self._current_iteration > self.max_iterations
        runtime_stop = time.perf_counter() - self._start_runtime > self.max_runtime

        return iter_stop or runtime_stop

import time
from alns.stop import StoppingCriterion

class NoImprovementTime(StoppingCriterion):
    def __init__(self, max_idle_seconds: float):
        self.max_idle_seconds = max_idle_seconds
        self.last_improvement_time = time.perf_counter()
        self.best_cost = float("inf")

    def __call__(self, rnd, best, current) -> bool:
        # Extrahiere Kosten aus State oder float
        best_val = best.objective() if hasattr(best, "objective") else best
        curr_val = current.objective() if hasattr(current, "objective") else current

        if best_val < self.best_cost:
            self.best_cost = best_val
            self.last_improvement_time = time.perf_counter()

        return (time.perf_counter() - self.last_improvement_time) > self.max_idle_seconds


from alns.stop import StoppingCriterion, NoImprovement, MaxRuntime

class OrStop(StoppingCriterion):
    def __init__(self, *criteria):
        self.criteria = criteria

    def __call__(self, rnd, best, current):
        # bricht ab, sobald eines der Kriterien True liefert
        return any(c(rnd, best, current) for c in self.criteria)
