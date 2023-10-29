from datetime import datetime
from typing import Optional, Generator
import configparser
import os

import numpy as np
from numpy.typing import NDArray
from netCDF4 import Dataset  # type: ignore

CONF_PARSER_TYPE = configparser.ConfigParser


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


def load_temps_dataset():
    import matplotlib.pyplot as plt

    with Dataset("NOAATemps_v5.1.0.nc", "r") as data:
        print(data.climatology[:])
        # print(data.variables)
        # print(data["anom"])
        # t = data.variables["time"][124 : 124 + 12 + 36]
        # q = np.average(data.variables["anom"][124 : 124 + 12 + 36, 0, :, :], axis=2)
        # print(q)
        # print(q.shape)
        # plt.plot(t, q[:, ::4])
        # plt.show()


def load_config(filename="DEFAULT.ini", path="OrbitalConstraints"):
    config = configparser.ConfigParser()
    config.read(path + "/" + filename)
    for sec in ["PDE", "PLANET", "ORBIT", "FILEMANAGEMENT"]:
        assert sec in config.sections()
    return config


def read_single_folder(foldername, folderpath):
    pass


def read_dual_folder(
    foldername, folderpath
) -> Generator[tuple[str, str, tuple[NDArray, NDArray, NDArray]], None, None]:
    dual, first_name, second_name = foldername.split("_")
    assert dual == "dual"  # must be a "dual..." folder
    os.chdir(folderpath + os.sep + foldername)
    files = os.listdir()
    for file in files:
        data = read_file(file)
        dual, first_name_file, first_val, second_name_file, second_val = file.strip(
            ".npz"
        ).split("_")
        yield first_val, second_val, data


if __name__ == "__main__":
    a = read_dual_folder("dual_a_e", os.path.curdir)
