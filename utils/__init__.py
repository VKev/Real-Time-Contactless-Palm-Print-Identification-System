
import os
import glob

modules = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))

__all__ = []

for module_path in modules:
    module_name = os.path.basename(module_path)[:-3]
    if module_name != "__init__":
        __all__.append(module_name)
        exec(f"from .{module_name} import *")