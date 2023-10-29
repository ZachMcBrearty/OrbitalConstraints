from datetime import datetime
from typing import Optional
import configparser
import os

import numpy as np
from netCDF4 import Dataset  # type: ignore

CONF_PARSER_TYPE = configparser.ConfigParser


def write_to_file(
    times: np.ndarray,
    temps: np.ndarray,
    degs: np.ndarray,
    filename: Optional[str] = None,
) -> None:
    if filename is None or filename == '""':
        filename = f"OrbCon{datetime.now().strftime('%Y%m%d%H%M%S')}.npz"
    np.savez(filename, times=times, temps=temps, degs=degs)


def read_files(filenames: str | list[str]) -> tuple:
    """fileanames: list of files to be loaded.
    returns: times, temps, degs"""
    if isinstance(filenames, str):
        with np.load(filenames) as a:
            times = a["times"]
            degs = a["degs"]
            temps = a["temps"]
    else:
        times = []
        temps = []
        degs = []
        for file_ in filenames:
            with np.load(file_) as a:
                times.append(a["times"])
                degs.append(a["degs"])
                temps.append(a["temps"])
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


def read_dual_folder(foldername, folderpath):
    dual, first_name, second_name = foldername.split("_")
    assert dual == "dual"  # must be a "dual..." folder
    os.chdir(folderpath + os.sep + foldername)
    files = os.listdir()
    data = read_files(files)
    first_val_range = []
    second_val_range = []
    for name in files:
        dual, first_name_file, first_val, second_name_file, second_val = name.split("_")
        if (
            dual != "dual"
            or first_name != first_name_file
            or second_name != second_name_file
        ):  # ingnore extraneous files
            continue
        first_val_range.append(first_val)
        second_val_range.append(second_val.strip(".npz"))
    return first_name, second_name, first_val_range, second_val_range, data


if __name__ == "__main__":
    fn, sn, fvr, svr, d = read_dual_folder("dual_a_e", os.path.curdir)
    print(fn, sn)
    print(fvr, svr)
