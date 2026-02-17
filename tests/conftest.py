import os
import tarfile
from pathlib import Path

import pytest
import ray
import requests

from pfb_imaging import set_envs

test_root_path = Path(__file__).resolve().parent
test_data_path = Path(test_root_path, "data")
test_data_path.mkdir(parents=True, exist_ok=True)

_data_tar_name = "test_ascii_1h60.0s.MS.tar.gz"
_ms_name = "test_ascii_1h60.0s.MS"

data_tar_path = Path(test_data_path, _data_tar_name)
ms_path = Path(test_data_path, _ms_name)

# https://drive.google.com/file/d/1rfGXGjjJ2XtF26LImlyJzCJMCNQZgEFT/view?usp=sharing

gdrive_id = "1rfGXGjjJ2XtF26LImlyJzCJMCNQZgEFT"

url = "https://drive.google.com/uc?id={id}".format(id=gdrive_id)


def pytest_sessionstart(session):
    """Called after Session object has been created, before run test loop."""

    if ms_path.exists():
        print("Test data already present - not downloading.")
    else:
        print("Test data not found - downloading...")
        download = requests.get(url)  # , params={"dl": 1}
        with open(data_tar_path, "wb") as f:
            f.write(download.content)
        with tarfile.open(data_tar_path, "r:gz") as tar:
            tar.extractall(path=test_data_path)
        data_tar_path.unlink()
        print("Test data successfully downloaded.")


@pytest.fixture(scope="session")
def ms_name():
    return str(ms_path)


@pytest.fixture(scope="session", autouse=True)
def manage_ray():
    def get_excludes():
        if os.path.exists(".rayignore"):
            return [line.strip() for line in open(".rayignore") if line.strip() and not line.startswith("#")]

    # Define the environment once
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    os.environ["PYTHONWARNINGS"] = "ignore:.*CUDA-enabled jaxlib is not installed.*"
    os.environ["RAY_PROCESS_SPAWN"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"

    env_vars = set_envs(2, 1)
    env_vars["JAX_LOGGING_LEVEL"] = "ERROR"
    env_vars["PYTHONWARNINGS"] = "ignore:.*CUDA-enabled jaxlib is not installed.*"
    env_vars["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    env_vars["RAY_RUNTIME_ENV_WORKING_DIR_MAX_SIZE_MB"] = "2048"

    runtime_env = {
        "env_vars": env_vars,
        "working_dir": None,
        "excludes": get_excludes(),
    }

    # Start Ray
    ray.init(num_cpus=1, runtime_env=runtime_env, ignore_reinit_error=True, include_dashboard=False)

    yield

    # Shutdown after all tests in the session are done
    ray.shutdown()
