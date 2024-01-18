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


def Habitable(temps: NDArray) -> NDArray:
    """aka BioCompatible
    temps: numpy array of temperatures.
    returns: 1 if 273 < temp < 373 ; 0 otherwise, for each temp in temps"""
    ret = np.ones_like(temps)
    ret[temps >= 373] = 0
    ret[temps <= 273] = 0
    return ret


def HumanCompatible(temps: NDArray) -> NDArray:
    """temps: numpy array of temperatures
    returns 0 if ANY T > 313K or ANY T < 263K; 1 if 273K < T < 303K; 0 otherwise, for each T in temps
    """
    ret = np.ones_like(temps)
    ret[temps >= 303] = 0  # 30 deg C
    ret[temps <= 273] = 0  # 0 deg C
    zeroes = np.zeros_like(temps[:, 0])
    for lat in range(temps.shape[1]):
        if np.any(temps[:, lat] >= 313) or np.any(temps[:, lat] <= 263):
            temps[:, lat] = zeroes
    return ret


def f_time(temps: floatarr, times: floatarr, dt: float, H=Habitable) -> floatarr:
    """temps: array of temperatures in space and time
    times: times corresponding to the temperatures
    dt: timestep
    H: habitability function; Habitable, HumanCompatible or user defined
    returns: time averaged habitablility for each latitude band"""
    H_temps = H(temps) * dt
    H_temps_sum = np.sum(H_temps, axis=0) / (times[-1] - times[0])
    return H_temps_sum


def f_area(temps: floatarr, lats: floatarr, dlat: float, H=Habitable) -> floatarr:
    """temps: array of temperatures in space and time
    lats: latitude bands corresponding to the temperatures
    dlat: latitude step size
    H: habitability function; Habitable, HumanCompatible or user defined
    returns: latitude averaged habitability for each time"""
    dx = np.cos(lats) * dlat
    H_temps = H(temps) * dx
    H_temps_sum = np.sum(H_temps, axis=1) / np.sum(dx)
    return H_temps_sum


def f_hab(
    temps: floatarr,
    lats: floatarr,
    dlat: float,
    times: floatarr,
    dt: float,
    H=Habitable,
) -> float:
    """temps: array of temperatures in space and time
    lats: latitude bands corresponding to the temperatures
    dlat: latitude step size
    times: times corresponding to the temperatures
    dt: timestep
    H: habitability function; Habitable, HumanCompatible or user defined
    returns: latitude and time averaged habitability"""
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
    min_year=90,
    max_year=100,
    H=Habitable,
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

        time_hab = f_time(temps_red, times_red, dt, H=H)
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
    min_year=90,
    max_year=95,
    H=Habitable,
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
        lat_hab = f_area(temps_red, lats, dlat, H=H)
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
    year=90,
    H=Habitable,
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
        tot_hab = f_hab(temps_red, lats, dlat, times_red, dt, H=H)
        if (q := float(val_1)) not in val_1_range:
            val_1_range.append(q)
        if (q := float(val_2)) not in val_2_range:
            val_2_range.append(q)
        total_habitability.append(tot_hab)
    xl = len(val_1_range)
    yl = len(val_2_range)
    val_1_range = np.array(val_1_range)
    val_2_range = np.array(val_2_range)
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

    k = 1 / (val_1_range[10] * (1 - val_2_range[0]))
    es = np.sqrt(1 - 1 / (val_1_range**4 * k))
    ax1.plot(val_1_range, es)

    # ax1.plot(xs, ys, c="m", label=r"$a \propto (1-e^2)^{-1/4}, T \approx 273$K")

    ax1.set_ylabel(f"{val_2_name} {val_2_unit}")
    ax1.set_xlabel(f"{val_1_name} {val_1_unit}")

    # ax1.legend(loc="lower right")
    plt.show()


def habitability_paramspace_compare(
    foldernames: tuple[str, str],
    folderpaths: tuple[str, str],
    val_1_name: str,
    val_2_name: str,
    val_1_unit: Optional[str] = None,
    val_2_unit: Optional[str] = None,
    year=90,
    H=Habitable,
):
    val_1_range_1 = []
    val_2_range_1 = []
    total_habitability = []
    data = read_dual_folder(foldernames[0], folderpaths[0])
    for i, datum in enumerate(data):
        val_1, val_2, (times, temps, degs) = datum
        lats = np.deg2rad(degs)
        dt = abs(times[1] - times[0])  # 1
        dlat = abs(lats[1] - lats[0])
        temps_red = temps.T[int(year / dt) : int((year + 1) / dt)]
        times_red = times[int(year / dt) : int((year + 1) / dt)]
        tot_hab = f_hab(temps_red, lats, dlat, times_red, dt, H=H)
        if (q := float(val_1)) not in val_1_range_1:
            val_1_range_1.append(q)
        if (q := float(val_2)) not in val_2_range_1:
            val_2_range_1.append(q)
        total_habitability.append(tot_hab)
    xl = len(val_1_range_1)
    yl = len(val_2_range_1)
    total_habitability_1 = np.array(total_habitability).reshape(xl, yl)

    val_1_range_2 = []
    val_2_range_2 = []
    total_habitability = []
    data = read_dual_folder(foldernames[1], folderpaths[1])
    for i, datum in enumerate(data):
        val_1, val_2, (times, temps, degs) = datum
        lats = np.deg2rad(degs)
        dt = abs(times[1] - times[0])  # 1
        dlat = abs(lats[1] - lats[0])
        temps_red = temps.T[int(year / dt) : int((year + 1) / dt)]
        times_red = times[int(year / dt) : int((year + 1) / dt)]
        tot_hab = f_hab(temps_red, lats, dlat, times_red, dt, H=H)
        if (q := float(val_1)) not in val_1_range_2:
            val_1_range_2.append(q)
        if (q := float(val_2)) not in val_2_range_2:
            val_2_range_2.append(q)
        total_habitability.append(tot_hab)
    xl = len(val_1_range_2)
    yl = len(val_2_range_2)
    total_habitability_2 = np.array(total_habitability).reshape(xl, yl)

    if val_1_unit is None:
        val_1_unit = ""
    else:
        val_1_unit = ", " + val_1_unit
    if val_2_unit is None:
        val_2_unit = ""
    else:
        val_2_unit = ", " + val_2_unit

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1: plt.Axes
    ax2: plt.Axes
    tot_hab_map = ax1.pcolormesh(
        val_1_range_1,
        val_2_range_1,
        total_habitability_1.T,
        cmap="RdBu_r",
        shading="nearest",
    )
    tot_hab_map = ax2.pcolormesh(
        val_1_range_2,
        val_2_range_2,
        total_habitability_2.T,
        cmap="RdBu_r",
        shading="nearest",
    )
    fig.colorbar(tot_hab_map, ax=ax1, label="Total Habitability")
    fig.colorbar(tot_hab_map, ax=ax2, label="Total Habitability")

    ax1.set_ylabel(f"{val_2_name} {val_2_unit}")
    ax1.set_xlabel(f"{val_1_name} {val_1_unit}")
    ax2.set_ylabel(f"{val_2_name} {val_2_unit}")
    ax2.set_xlabel(f"{val_1_name} {val_1_unit}")

    plt.show()


if __name__ == "__main__":
    import os

    here = os.path.curdir

    # habitability_paramspace_compare(
    #     (
    #         "dual_gassemimajoraxis_gaseccentricity_TH_0",
    #         "dual_gassemimajoraxis_gaseccentricity_TH_0.003_0.005",
    #     ),
    #     (here, here),
    #     "a$_{moon}$",
    #     "e$_{moon}$",
    #     a_unit,
    #     e_unit,
    #     H=Habitable,
    # )

    # habitability_paramspace(
    #     "dual_gassemimajoraxis_gaseccentricity_TH_0.003_0.007",
    #     os.path.curdir,
    #     "a$_{moon}$",
    #     "e$_{moon}$",
    #     a_unit,
    #     e_unit,
    #     H=Habitable,
    # )
    # habitability_paramspace(
    #     "dual_moonsemimajoraxis_mooneccentricity",
    #     os.path.curdir,
    #     "a$_{moon}$",
    #     "e$_{moon}$",
    #     a_unit,
    #     e_unit,
    #     H=Habitable,
    # )
    # time_habitability_paramspace("single_a", os.path.curdir, "a", a_unit)
    # time_habitability_paramspace(
    #     "single_gassemimajoraxis_TH_0.003_0.008",
    #     os.path.curdir,
    #     "a",
    #     a_unit,
    #     H=HumanCompatible,
    # )
    # area_habitability_paramspace(
    #     "single_gassemimajoraxis_TH_0.003_0.008",
    #     os.path.curdir,
    #     "a",
    #     a_unit,
    #     H=HumanCompatible,
    # )
    # time_habitability_paramspace(
    #     "single_moonsemimajoraxis", os.path.curdir, "a", a_unit, H=BioCompatible
    # )
    # time_habitability_paramspace("single_e", os.path.curdir, "e", e_unit)
    # time_habitability_paramspace("single_gaseccentricity", os.path.curdir, "e", e_unit)
    # time_habitability_paramspace(
    #     "single_mooneccentricity", os.path.curdir, "e", e_unit, H=BioCompatible
    # )
    # time_habitability_paramspace(
    #     "single_obliquity", os.path.curdir, "obliquity", obliquity_unit
    # )
    # area_habitability_paramspace(
    #     "single_obliquity", os.path.curdir, "obliquity", obliquity_unit
    # )
    # time_habitability_paramspace("single_omega", os.path.curdir, "omega", omega_unit)
    # time_habitability_paramspace(
    #     "single_starttemp", os.path.curdir, "starttemp", temp_unit
    # )
    # time_habitability_paramspace(
    #     "single_timestep", os.path.curdir, "timestep", timestep_unit
    # )

    # area_habitability_paramspace("single_a", os.path.curdir, "a", a_unit)
    # area_habitability_paramspace("single_gassemimajoraxis", os.path.curdir, "a", a_unit)
    # area_habitability_paramspace("single_e", os.path.curdir, "e", e_unit)
    # area_habitability_paramspace("single_gaseccentricity", os.path.curdir, "e", e_unit)

    # area_habitability_paramspace("single_omega", os.path.curdir, "omega", omega_unit)
    # area_habitability_paramspace(
    #     "single_starttemp", os.path.curdir, "starttemp", temp_unit
    # )
    # area_habitability_paramspace(
    #     "single_spacedim", os.path.curdir, "spacedim", spacedim_unit
    # )

    # habitability_paramspace("dual_a_e", os.path.curdir, "a", "e", a_unit, e_unit)
    habitability_paramspace(
        "dual_gassemimajoraxis_gaseccentricity",
        os.path.curdir,
        "a$_{gas}$",
        "e$_{gas}$",
        a_unit,
        e_unit,
        year=90,
        H=HumanCompatible,
    )
    # habitability_paramspace(
    #     "dual_moonsemimajoraxis_mooneccentricity",
    #     os.path.curdir,
    #     "a$_{moon}$",
    #     "e$_{moon}$",
    #     a_unit,
    #     e_unit,
    #     year=90,
    # )
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
