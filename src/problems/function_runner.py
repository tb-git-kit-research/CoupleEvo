import threading
import time
import traceback

# ----------------------------
# Threaded function runner
# ----------------------------
class FunctionRunner:
    def __init__(self, func, index=None, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.index = index
        self._thread = None
        self._stop_flag = threading.Event()
        self.result = None
        self.exception = None

    def _target(self):
        try:
            self.result = self.func(*self.args, stop_flag=self._stop_flag, **self.kwargs)
        except Exception as e:
            self.exception = e
            traceback.print_exc()

    def start(self):
        self._thread = threading.Thread(target=self._target, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()

    def join(self, timeout=None):
        self._thread.join(timeout)
        return not self._thread.is_alive()  # True if finished
