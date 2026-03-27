

import multiprocessing as mp
import traceback

# ----------------------------
# Generic process wrapper
# ----------------------------
class ProcessRunner:
    def __init__(self, func, index=None, *args, **kwargs):
        """
        Generic process runner for running functions safely with timeout.

        Args:
            func (callable): The function to run inside a subprocess.
            index (int): Task index to preserve order.
            *args, **kwargs: Arguments passed to the function.
        """
        self.func = func
        self.index = index
        self.args = args
        self.kwargs = kwargs
        self.result = mp.Manager().dict()
        self._process = None

    def _target(self, result_dict):
        try:
            output = self.func(*self.args, **self.kwargs)
            result_dict["result"] = output
        except Exception:
            traceback.print_exc()
            result_dict["result"] = None

    def start(self):
        self._process = mp.Process(target=self._target, args=(self.result,))
        self._process.start()

    def join(self, timeout=None):
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
            return False  # Not finished within timeout
        return True

    @property
    def output(self):
        return self.result.get("result", None)
