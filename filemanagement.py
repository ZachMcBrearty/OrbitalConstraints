from datetime import datetime
from typing import Optional, Generator
import configparser
import os

import numpy as np
from numpy.typing import NDArray


def write_to_file(
    times: NDArray,
    temps: NDArray,
    degs: NDArray,
    filename: Optional[str] = None,
) -> None:
    if filename is None or filename == '""':
        filename = f"OrbCon{datetime.now().strftime('%Y%m%d%H%M%S')}.npz"
    np.savez(filename, times=times, temps=temps, degs=degs)


def read_file(filename: str) -> tuple[NDArray, NDArray, NDArray]:
    """filenames: file to be loaded.
    returns: (times, temps, degs)
    """
    with np.load(filename) as a:
        times = a["times"]
        degs = a["degs"]
        temps = a["temps"]
    return times, temps, degs


def load_config(filename="DEFAULT.ini", path="OrbitalConstraints"):
    config = configparser.ConfigParser()
    config.read(path + "/" + filename)
    for sec in ["PDE", "PLANET", "ORBIT", "FILEMANAGEMENT"]:
        assert sec in config.sections()
    return config


def read_single_folder(
    foldername: str, folderpath: str
) -> Generator[tuple[str, tuple[NDArray, NDArray, NDArray]], None, None]:
    single, first_name, *extra = foldername.split("_")
    assert single == "single"  # must be a "single..." folder
    files = os.listdir(folderpath + os.sep + foldername)
    files = sorted(files, key=lambda x: float(x.strip(".npz").split("_")[-1]))
    for file in files:
        data = read_file(folderpath + os.sep + foldername + os.sep + file)
        single, *first_name_file, first_val = file.strip(".npz").split("_")
        yield first_val, data


def read_dual_folder(
    foldername: str, folderpath: str
) -> Generator[tuple[str, str, tuple[NDArray, NDArray, NDArray]], None, None]:
    dual, first_name, second_name, *extra = foldername.split("_")
    assert dual == "dual"  # must be a "dual..." folder
    files = os.listdir(folderpath + os.sep + foldername)
    files = sorted(files, key=lambda x: float(x.strip(".npz").split("_")[4]))
    files = sorted(files, key=lambda x: float(x.strip(".npz").split("_")[2]))
    for file in files:
        data = read_file(folderpath + os.sep + foldername + os.sep + file)
        dual, first_name_file, first_val, second_name_file, second_val = file.strip(
            ".npz"
        ).split("_")
        yield first_val, second_val, data


if __name__ == "__main__":
    a = read_dual_folder("dual_a_e", os.path.curdir)
