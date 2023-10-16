from datetime import datetime

import numpy as np
from netCDF4 import Dataset  # type: ignore

from typing import Optional


def write_to_file(
    times: np.ndarray,
    temps: np.ndarray,
    degs: np.ndarray,
    filename: Optional[str] = None,
) -> None:
    if filename is None:
        filename = f"OrbCon{datetime.now().strftime('%Y%m%d%H%M%S')}.npz"
    np.savez(filename, times=times, temps=temps, degs=degs)


def read_files(filenames: list[str]) -> tuple:
    """fileanames: list of files to be loaded.
    returns: times, degs, temps"""
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
    with Dataset("NOAATemps.nc", "r") as data:
        print(data.variables)
        # print(data["anom"])
        # q = np.average(data.variables["anom"][124 : 124 + 12, 0, :, :], axis=2)
        # print(q + 13.9)
        # print(q.shape)


if __name__ == "__main__":
    # times = np.array([0, 2, 4, 6, 8])
    # degs = np.array([1, 2, 3, 4])
    # temps = np.array(
    #     [[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16], [5, 10, 15, 20]]
    # )
    # write_to_file(times, degs, temps, filename="one.npz")
    # write_to_file(times * 2, degs * 2, temps * 2, filename="two.npz")
    # write_to_file(times * 4, degs * 4, temps * 4, filename="three.npz")
    # print(read_files(["one.npz", "two.npz", "three.npz"]))
    load_temps_dataset()
