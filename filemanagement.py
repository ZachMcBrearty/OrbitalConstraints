from datetime import datetime

import numpy as np

from typing import Optional


def write_to_file(
    times: np.ndarray,
    temps: np.ndarray,
    degs: np.ndarray,
    filename: Optional[str] = None,
) -> None:
    if filename is None:
        filename = f"OrbCon{datetime.now().strftime('%Y%m%d%H%M%S')}.npz"
    np.savez(filename, times=times, degs=degs, temps=temps)


def read_files(filenames: list[str]) -> tuple:
    times = []
    degs = []
    temps = []
    for file_ in filenames:
        with np.load(file_) as a:
            times.append(a["times"])
            degs.append(a["degs"])
            temps.append(a["temps"])
    return times, degs, temps


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
