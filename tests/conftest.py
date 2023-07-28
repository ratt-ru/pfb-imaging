import pytest
from pathlib import Path
import requests
import tarfile

test_root_path = Path(__file__).resolve().parent
test_data_path = Path(test_root_path, "data")
test_data_path.mkdir(parents=True, exist_ok=True)

_data_tar_name = "test_ascii_1h60.0s.MS.tar.gz"
_ms_name = "test_ascii_1h60.0s.MS"

data_tar_path = Path(test_data_path, _data_tar_name)
ms_path = Path(test_data_path, _ms_name)

#https://drive.google.com/file/d/1rfGXGjjJ2XtF26LImlyJzCJMCNQZgEFT/view?usp=sharing

gdrive_id = "1rfGXGjjJ2XtF26LImlyJzCJMCNQZgEFT"

url = "https://drive.google.com/uc?id={id}".format(id=gdrive_id)

def pytest_sessionstart(session):
    """Called after Session object has been created, before run test loop."""

    if ms_path.exists():
        print("Test data already present - not downloading.")
    else:
        print("Test data not found - downloading...")
        download = requests.get(url)  # , params={"dl": 1}
        with open(data_tar_path, 'wb') as f:
            f.write(download.content)
        with tarfile.open(data_tar_path, "r:gz") as tar:
            tar.extractall(path=test_data_path)
        data_tar_path.unlink()
        print("Test data successfully downloaded.")


@pytest.fixture(scope='session')
def ms_name():
    return str(ms_path)
