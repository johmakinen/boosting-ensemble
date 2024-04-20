import cProfile
import pstats
import sys
import os
from pathlib import Path
from contextlib import redirect_stdout

sys.path.append(str(Path(__file__).parents[1]))
from src.models import EnsembleModel
from src.configs import config, config_test
from src.utils import classification_data, regression_data

global_conf = config.copy()  # or config_test.copy()


def get_model(task: str = "classification"):
    if task == "classification":
        data = classification_data()
    else:
        data = regression_data()
    model = EnsembleModel(config=global_conf, task=task, datasets_=data)
    model.train()
    model.evaluate(plot=False)
    return model


if __name__ == "__main__":

    # # Get directory of this file
    dir_ = Path(__file__).parent / "profiles"
    dir_.mkdir(exist_ok=True)
    with redirect_stdout(open(dir_ / f"{get_model.__name__}_cprof.txt", "w")):
        cProfile.run(f"{get_model.__name__}(task='classification')", sort="tottime")


# cprofile
# python profiling/profiling.py
# OR
# python -m profiling.profiling
