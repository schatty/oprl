import sys
import importlib.util


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config 
    spec.loader.exec_module(config)
    return config
