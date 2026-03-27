import importlib.util

file_path = "src/libraries/lns/wrapper/cvrp.py"  # update with your real path

spec = importlib.util.spec_from_file_location("cvrp_module", file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

cls = getattr(module, "ProblemState")
print(cls)
print(cls.__module__)
print(cls.__name__)

print(cls.__module__.__file__)

import inspect
try:
    print(inspect.getfile(cls))
except Exception as e:
    print("getfile error:", e)