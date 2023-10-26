from datetime import datetime
from typing import Optional
import configparser

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


if __name__ == "__main__":
    times = np.array([0, 2, 4, 6, 8])
    degs = np.array([1, 2, 3, 4])
    temps = np.array(
        [[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16], [5, 10, 15, 20]]
    )
    write_to_file(times, degs, temps, filename="one.npz")
    write_to_file(times * 2, degs * 2, temps * 2, filename="two.npz")
    write_to_file(times * 4, degs * 4, temps * 4, filename="three.npz")
    print(read_files(["one.npz", "two.npz", "three.npz"]))
