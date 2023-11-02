# SMS08 - 3.2 "fractional habitability"
# Habitable if temperature is between 273K and 373K (0-100C)
# Habitabilty function: H[a, λ, t] = {1 if 273 < t < 273 ; 0 otherwise}

# -> Temporal habitability: f_time[a, λ] = int_0^P(H[a,λ,t] dt) / P
# i.e. integrate over a period P to find if [a,λ] is habitable throughout

# -> Planetary habitability: f_area[a, t] = int_{-pi/2}^{pi/2}(H[a,λ,t] cosλ dλ) / 2
# i.e. integrate over all latitudes to find if [a, t] is habitable

# => combine both to get the net fractional habitability, weighted by area
# f_hab[a] = int_{-pi/2}^{pi/2}( int_0^P(H[a,λ,t] dt) cosλ dλ) / 2P
# fraction of λ-t plane which is habitable

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from Constants import *
from filemanagement import read_dual_folder, read_single_folder, read_file


def H(temps: NDArray) -> NDArray:
    """temps: numpy array of temperatures.
    returns: 1 if 273 < temp < 373 ; 0 otherwise, for each temp in temps"""
    ret = np.ones_like(temps)
    ret[temps >= 373] = 0
    ret[temps <= 273] = 0
    return ret


def f_time(temps: floatarr, times: floatarr, dt: float) -> floatarr:
    H_temps = H(temps) * dt
    H_temps_sum = np.sum(H_temps, axis=0) / (times[-1] - times[0])
    return H_temps_sum


def f_area(temps: floatarr, lats: floatarr, dlat: float) -> floatarr:
    dx = np.cos(lats) * dlat
    H_temps = H(temps) * dx
    H_temps_sum = np.sum(H_temps, axis=1) / np.sum(dx)
    return H_temps_sum


def f_hab(
    temps: floatarr, lats: floatarr, dlat: float, times: floatarr, dt: float
) -> float:
    dx = np.cos(lats) * dlat
    H_temps = H(temps) * dt * dx
    H_temps_sum = np.sum(H_temps, axis=1) / np.sum(dx)
    H_temps_sum_sum = np.sum(H_temps_sum, axis=0) / (times[-1] - times[0])
    return H_temps_sum_sum


def time_habitability_paramspace(
    foldername: str,
    folderpath: str,
    val_name: str,
    val_unit: Optional[str] = None,
    min_year=190,
    max_year=200,
):
    """Latitude band habitability, averaged over a number of years
    min_year to max_year: years to average over"""
    assert val_name != "spacedim"  # i.e. spacedim should be constant
    val_range = []
    habitability_time = []
    data = read_single_folder(foldername, folderpath)
    degs = None
    for i, datum in enumerate(data):
        val, (times, temps, degs) = datum
        dt = abs(times[1] - times[0])  # years
        # 365 * dt years = dt days
        # 365 days per year / (365 * dt) days = datapoints per year
        # datapoints per year = 1 / dt
        slice_min = int(min_year / dt)
        slice_max = int((max_year + 1) / dt)
        temps_red = temps.T[slice_min:slice_max]
        times_red = times[slice_min:slice_max]

        time_hab = f_time(temps_red, times_red, dt)
        if (float_val := float(val)) not in val_range:
            val_range.append(float_val)
        habitability_time.append(time_hab)

    assert degs is not None

    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit

    fig, ax = plt.subplots(1, 1)
    ax: plt.Axes
    time_hab_map = ax.pcolormesh(
        val_range, degs, np.array(habitability_time).T, cmap="RdBu_r", shading="nearest"
    )
    fig.colorbar(time_hab_map, ax=ax, label="Time averaged habitability")
    ax.set_ylabel(r"Latitude, $^{\circ}$")
    ax.set_xlabel(f"{val_name} {val_unit}")
    plt.show()


def area_habitability_paramspace(
    foldername: str,
    folderpath: str,
    val_name: str,
    val_unit: Optional[str] = None,
    min_year=190,
    max_year=195,
):
    """yearly habitability averaged over all latitudes
    min year to max_year: years to show"""
    assert val_name != "timestep"  # i.e. timestep should be constant
    val_range = []
    habitability_lat = []
    data = read_single_folder(foldername, folderpath)
    times_red = None
    for i, datum in enumerate(data):
        val, (times, temps, degs) = datum
        lats = np.deg2rad(degs)
        dlat = abs(lats[1] - lats[0])
        dt = abs(times[1] - times[0])  # years
        # 365 * dt years = dt days
        # 365 days per year / (365 * dt) days = datapoints per year
        # datapoints per year = 1 / dt
        slice_min = int(min_year / dt)
        slice_max = int((max_year) / dt) + 1
        temps_red = temps.T[slice_min:slice_max]
        times_red = times[slice_min:slice_max]
        lat_hab = f_area(temps_red, lats, dlat)
        if (float_val := float(val)) not in val_range:
            val_range.append(float_val)
        habitability_lat.append(lat_hab)

    assert times_red is not None

    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit

    fig, ax = plt.subplots(1, 1)
    ax: plt.Axes
    lat_hab_map = ax.pcolormesh(
        val_range,
        times_red,
        np.array(habitability_lat).T,
        cmap="RdBu_r",
        shading="nearest",
    )
    fig.colorbar(lat_hab_map, ax=ax, label="Area averaged habitability")
    ax.set_ylabel("Time, years")
    ax.set_xlabel(f"{val_name} {val_unit}")
    plt.show()


def habitability_paramspace(
    foldername: str,
    folderpath: str,
    val_1_name: str,
    val_2_name: str,
    val_1_unit: Optional[str] = None,
    val_2_unit: Optional[str] = None,
    year=190,
):
    val_1_range = []
    val_2_range = []
    total_habitability = []
    data = read_dual_folder(foldername, folderpath)
    for i, datum in enumerate(data):
        val_1, val_2, (times, temps, degs) = datum
        lats = np.deg2rad(degs)
        dt = abs(times[1] - times[0])  # 1
        dlat = abs(lats[1] - lats[0])
        temps_red = temps.T[int(year / dt) : int((year + 1) / dt)]
        times_red = times[int(year / dt) : int((year + 1) / dt)]
        tot_hab = f_hab(
            temps_red,
            lats,
            dlat,
            times_red,
            dt,
        )
        if val_1 not in val_1_range:
            val_1_range.append(val_1)
        if val_2 not in val_2_range:
            val_2_range.append(val_2)
        total_habitability.append(tot_hab)
    xl = len(val_1_range)
    yl = len(val_2_range)
    total_habitability = np.array(total_habitability).reshape(xl, yl)

    if val_1_unit is None:
        val_1_unit = ""
    else:
        val_1_unit = ", " + val_1_unit
    if val_2_unit is None:
        val_2_unit = ""
    else:
        val_2_unit = ", " + val_2_unit

    fig, ax1 = plt.subplots(1, 1)
    ax1: plt.Axes
    tot_hab_map = ax1.pcolormesh(
        val_1_range, val_2_range, total_habitability.T, cmap="RdBu_r", shading="nearest"
    )
    fig.colorbar(tot_hab_map, ax=ax1, label="Total Habitability")
    ax1.set_ylabel(f"{val_2_name} {val_2_unit}")
    ax1.set_xlabel(f"{val_1_name} {val_1_unit}")
    plt.show()


if __name__ == "__main__":
    import os

    # time_habitability_paramspace("single_a", os.path.curdir, "a", a_unit)
    # time_habitability_paramspace("single_e", os.path.curdir, "e", e_unit)
    # time_habitability_paramspace(
    #     "single_obliquity", os.path.curdir, "obliquity", obliquity_unit
    # )
    time_habitability_paramspace("single_omega", os.path.curdir, "omega", omega_unit)
    # time_habitability_paramspace(
    #     "single_starttemp", os.path.curdir, "starttemp", temp_unit
    # )
    # time_habitability_paramspace(
    #     "single_timestep", os.path.curdir, "timestep", timestep_unit
    # )

    # area_habitability_paramspace("single_a", os.path.curdir, "a", a_unit)
    # area_habitability_paramspace("single_e", os.path.curdir, "e", e_unit)
    # area_habitability_paramspace(
    #     "single_obliquity", os.path.curdir, "obliquity", obliquity_unit
    # )
    area_habitability_paramspace("single_omega", os.path.curdir, "omega", omega_unit)
    # area_habitability_paramspace(
    #     "single_starttemp", os.path.curdir, "starttemp", temp_unit
    # )
    # area_habitability_paramspace(
    #     "single_spacedim", os.path.curdir, "spacedim", spacedim_unit
    # )

    # habitability_paramspace("dual_a_e", os.path.curdir, "a", "e", a_unit, e_unit)
    # habitability_paramspace(
    #     "dual_a_obliquity", os.path.curdir, "a", "obliquity", a_unit, obliquity_unit
    # )
    # habitability_paramspace(
    #     "dual_a_omega", os.path.curdir, "a", "omega", a_unit, omega_unit
    # )
    # habitability_paramspace(
    #     "dual_a_starttemp", os.path.curdir, "a", "starttemp", a_unit, temp_unit
    # )

    # habitability_paramspace(
    #     "dual_e_obliquity", os.path.curdir, "e", "obliquity", e_unit, obliquity_unit
    # )
    # habitability_paramspace(
    #     "dual_e_omega", os.path.curdir, "e", "omega", e_unit, omega_unit
    # )
    # habitability_paramspace(
    #     "dual_e_starttemp", os.path.curdir, "e", "starttemp", e_unit, temp_unit
    # )

    # habitability_paramspace(
    #     "dual_obliquity_omega",
    #     os.path.curdir,
    #     "obliquity",
    #     "omega",
    #     obliquity_unit,
    #     omega_unit,
    # )
    # habitability_paramspace(
    #     "dual_obliquity_starttemp",
    #     os.path.curdir,
    #     "obliquity",
    #     "starttemp",
    #     obliquity_unit,
    #     temp_unit,
    # )

    # habitability_paramspace(
    #     "dual_omega_starttemp",
    #     os.path.curdir,
    #     "omega",
    #     "starttemp",
    #     omega_unit,
    #     temp_unit,
    # )

    # times, temps, degs = read_file("testing_3.npz")
    # times = times[365 * 150 : 365 * 151 + 1]
    # temps = temps.T[365 * 150 : 365 * 151 + 1]
    # print(times.shape)
    # print(temps.shape)
    # degs = degs
    # lats = np.deg2rad(degs)

    # dt = abs(times[1] - times[0])
    # dlat = abs(degs[1] - degs[0])

    # fig, (ax1, ax2) = plt.subplots(2, 1)

    # ax1.plot(degs, f_time(temps, times, dt))
    # ax1.set_xlabel("Latitude")
    # ax1.set_ylabel("f_time")
    # ax2.plot(times, f_area(temps, lats, dlat))
    # ax2.set_xlabel("Time")
    # ax2.set_ylabel("f_area")
    # print(f_hab(temps, lats, dlat, times, dt))

    # plt.show()
