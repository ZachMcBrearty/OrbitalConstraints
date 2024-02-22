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


def Habitable(temps: NDArray, maxtol=100) -> NDArray:
    """aka BioCompatible
    temps: numpy array of temperatures.
    returns: 1 if 273 < temp < 373 ; 0 otherwise, for each temp in temps"""
    ret = np.ones_like(temps)
    ret[temps >= 373] = 0
    ret[temps <= 273] = 0
    zeroes = np.zeros_like(temps[:, 0])
    for lat in range(temps.shape[1]):
        if np.any(temps[:, lat] >= 373 + maxtol) or np.any(
            temps[:, lat] <= 273 - maxtol
        ):
            ret[:, lat] = zeroes
    return ret


def HumanCompatible(temps: NDArray, maxtol=10) -> NDArray:
    """temps: numpy array of temperatures
    returns 0 if ANY T > 313K or ANY T < 263K; 1 if 273K < T < 303K; 0 otherwise, for each T in temps
    """
    ret = np.ones_like(temps)
    ret[temps >= 303] = 0  # 30 deg C
    ret[temps <= 273] = 0  # 0 deg C
    zeroes = np.zeros_like(temps[:, 0])
    for lat in range(temps.shape[1]):
        if np.any(temps[:, lat] >= 303 + maxtol) or np.any(
            temps[:, lat] <= 273 - maxtol
        ):
            ret[:, lat] = zeroes
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


def habitabilitycolourplot(
    degs,
    temps,
    times,
    yr_start=None,
    yr_end=None,
    # year_avg=1,
    lat_start=None,
    lat_end=None,
    H=Habitable,
    H_label="",
):
    if yr_start is None:
        yr_start = 0
    else:
        yr_start *= 365
    if yr_end is None:
        yr_end = temps.shape[1]
    else:
        yr_end = (yr_end + 1) * 365

    if lat_start is None:
        lat_start = 0
    if lat_end is None:
        lat_end = degs.shape[0]

    fig, ax = plt.subplots(1, 1)
    cmap = "RdBu_r"
    ts = times[yr_start:yr_end]
    temp = temps[lat_start:lat_end, yr_start + 1 : yr_end + 1]
    habs = H(temp)
    # print(np.sum(habs))
    pcm = ax.pcolormesh(ts, degs[lat_start:lat_end], habs, cmap=cmap, shading="nearest")

    ax.set_xlabel("time, yr")
    ax.set_ylabel("latitude, degrees")
    ax.set_yticks(np.linspace(degs[lat_start], degs[lat_end - 1], 13, endpoint=True))
    fig.colorbar(pcm, ax=ax, label="Total Habitability " + H_label)
    plt.tight_layout()
    plt.show()


def time_habitability_paramspace(
    foldername: str,
    folderpath: str,
    val_name: str,
    val_unit: Optional[str] = None,
    min_year=90,
    max_year=100,
    H=Habitable,
    H_label="",
    vertical_lines: Optional[list[tuple[float, str, str]]] = None,
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
        val_range,
        degs,
        np.array(habitability_time).T,
        cmap="viridis",
        shading="nearest",
    )
    fig.colorbar(time_hab_map, ax=ax, label="Time averaged habitability " + H_label)
    ax.set_yticks(np.linspace(-90, 90, 11))
    # ax.set_xticks(np.linspace(0, 90, 11))
    if vertical_lines is not None:
        for line in vertical_lines:
            ax.axvline(line[0], 0, 1, ls=line[1], label=line[2], c="r")
        plt.legend()
    ax.set_ylabel(r"Latitude, $\lambda$, $^{\circ}$")
    ax.set_xlabel(f"{val_name} {val_unit}")
    plt.tight_layout()
    plt.show()


def area_habitability_paramspace(
    foldername: str,
    folderpath: str,
    val_name: str,
    val_unit: Optional[str] = None,
    min_year=90,
    max_year=95,
    H=Habitable,
    H_label="",
    vertical_lines: Optional[list[tuple[float, str, str]]] = None,
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
        cmap="RdBu",
        shading="nearest",
    )
    fig.colorbar(lat_hab_map, ax=ax, label="Area averaged habitability " + H_label)
    # ax.set_xticks(np.linspace(val_range[0], val_range[-1], 11))
    if vertical_lines is not None:
        for line in vertical_lines:
            ax.axvline(line[0], 0, 1, ls=line[1], label=line[2], c="g")
        plt.legend()
    ax.set_ylabel("Time, t, years")
    ax.set_xlabel(f"{val_name} {val_unit}")
    plt.tight_layout()
    plt.show()


def time_and_area_shared_paramspace(
    foldername: str,
    folderpath: str,
    val_name: str,
    val_unit: Optional[str] = None,
    min_year=90,
    max_year=100,
    H=Habitable,
):
    val_range = []
    habitability_lat = []
    habitability_time = []
    data = read_single_folder(foldername, folderpath)
    degs = None
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
        time_hab = f_time(temps_red, times_red, dt, H=H)
        if (float_val := float(val)) not in val_range:
            val_range.append(float_val)
        habitability_lat.append(lat_hab)
        habitability_time.append(time_hab)
    assert degs is not None
    assert times_red is not None

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1: plt.Axes
    ax2: plt.Axes
    time_hab_map = ax1.pcolormesh(
        val_range, degs, np.array(habitability_time).T, cmap="RdBu_r", shading="nearest"
    )
    fig.colorbar(time_hab_map, ax=ax1, label="Time averaged habitability")
    ax1.set_xticks(np.linspace(val_range[0], val_range[-1], 11))
    ax1.set_yticks(np.arange(-90, 91, 30))
    ax1.set_ylabel(r"Latitude, $\lambda$, $^{\circ}$")
    # ax1.set_xlabel(f"{val_name} {val_unit}")

    lat_hab_map = ax2.pcolormesh(
        val_range,
        times_red,
        np.array(habitability_lat).T,
        cmap="RdBu_r",
        shading="nearest",
    )
    fig.colorbar(lat_hab_map, ax=ax2, label="Area averaged habitability")
    ax2.set_xticks(np.linspace(val_range[0], val_range[-1], 11))
    ax2.axvline(DISTANCE["venus"] / AU, 0, 1, ls="dashed", label="Venus")
    ax2.axvline(DISTANCE["mars"] / AU, 0, 1, ls="dashdot", label="Mars")
    # ax.axvline(22, 0, 1, ls="dashed", label="Earth min/max")
    # ax.axvline(25, 0, 1, ls="dashed")
    ax2.set_ylabel("Time, t, years")
    ax2.set_xlabel(f"{val_name} {val_unit}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def habitability_paramspace(
    foldername: str,
    folderpath: str,
    val_1_name: str,
    val_2_name: str,
    val_1_unit: Optional[str] = None,
    val_2_unit: Optional[str] = None,
    yearstart=90,
    yearlength=3,
    H=Habitable,
    Hlabel="",
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
        temps_red = temps.T[int(yearstart / dt) : int((yearstart + yearlength) / dt)]
        times_red = times[int(yearstart / dt) : int((yearstart + yearlength) / dt)]
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
    fig.colorbar(tot_hab_map, ax=ax1, label="Total Habitability" + Hlabel)

    # k = 1 / (val_1_range[10] * (1 - val_2_range[0]))
    # es = np.sqrt(1 - 1 / (val_1_range**4 * k))
    # ax1.plot(val_1_range, es)

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
    # here = "D:/v2/"
    from filemanagement import read_file

    # times, temps, degs = read_file("Earth.npz")
    # habitabilitycolourplot(
    #     degs, temps, times, 180, 182, None, None, HumanCompatible, "(HC)"
    # )
    # habitabilitycolourplot(degs, temps, times, 180, 182, None, None, Habitable, "(LWR)")

    # lats = np.deg2rad(degs)
    # print(times.shape, temps.shape, degs.shape)
    # timehab = f_time(
    #     temps.T[365 * 190 : 365 * 193],
    #     times[365 * 190 : 365 * 193],
    #     times[1] - times[0],
    #     HumanCompatible,
    # )
    # spacehab = f_area(
    #     temps.T[365 * 190 : 365 * 193], lats, lats[1] - lats[0], HumanCompatible
    # )
    # print(
    #     f_hab(
    #         temps.T[365 * 190 : 365 * 193],
    #         lats,
    #         lats[1] - lats[0],
    #         times[365 * 190 : 365 * 193],
    #         times[1] - times[0],
    #         HumanCompatible,
    #     )
    # )
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1: plt.Axes
    # ax2: plt.Axes
    # ax1.scatter(degs, timehab)
    # ax1.set_xlabel(r"Latitude, degrees")
    # ax1.set_ylabel(r"Time averaged habitability (LWR), $f_{time}$")
    # ax1.set_xticks(np.linspace(-90, 90, 7))

    # ax2.scatter(
    #     times[365 * 190 : 365 * 193],
    #     spacehab,
    # )
    # ax2.set_xlabel(r"Time, yr")
    # ax2.set_ylabel(r"Area averaged habitability (LWR), $f_{area}$")
    # # plt.gca().set_aspect(3)
    # plt.show()

    # here = "D:/v2"

    # time_and_area_shared_paramspace(
    #     "single_obliquity", here, obliquity_name, obliquity_unit, 150, 160, H=Habitable
    # )
    # time_and_area_shared_paramspace(
    #     "single_obliquity",
    #     here,
    #     obliquity_name,
    #     obliquity_unit,
    #     150,
    #     160,
    #     H=HumanCompatible,
    # )
    suff = ""  # "_TH_0.003_0.006"
    # area_habitability_paramspace(
    #     "single_gassemimajoraxis" + suff,
    #     here,
    #     agas_name,
    #     a_unit,
    #     180,
    #     182,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    #     vertical_lines=[
    #         (DISTANCE["venus"] / AU, "dashed", "Venus"),
    #         (1.5, "dashdot", "Mars"),
    #     ],
    # )
    # time_habitability_paramspace(
    #     "single_gassemimajoraxis" + suff,
    #     here,
    #     agas_name,
    #     a_unit,
    #     180,
    #     200,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    #     vertical_lines=[
    #         (DISTANCE["venus"] / AU, "dashed", "Venus"),
    #         (1.5, "dashdot", "Mars"),
    #     ],
    # )
    # area_habitability_paramspace(
    #     "single_gaseccentricity" + suff,
    #     here,
    #     egas_name,
    #     e_unit,
    #     180,
    #     182,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    #     # vertical_lines=[
    #     #     (EARTH_CURRENT_ECCENTRICITY, "dashed", "Earth Current"),
    #     #     (EARTH_MAX_ECCENTRICITY, "dashdot", "Earth Max"),
    #     # ],
    # )
    # time_habitability_paramspace(
    #     "single_gaseccentricity" + suff,
    #     here,
    #     egas_name,
    #     e_unit,
    #     180,
    #     200,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    #     # vertical_lines=[
    #     #     (EARTH_CURRENT_ECCENTRICITY, "dashed", "Earth Current"),
    #     #     (EARTH_MAX_ECCENTRICITY, "dashdot", "Earth Max"),
    #     # ],
    # )
    # area_habitability_paramspace(
    #     "single_obliquity" + suff,
    #     here,
    #     obliquity_name,
    #     obliquity_unit,
    #     180,
    #     182,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    #     # vertical_lines=[
    #     #     (EARTH_MIN_OBLIQUITY, "dashed", "Earth Min"),
    #     #     (EARTH_MAX_OBLIQUITY, "dashdot", "Earth Max"),
    #     # ],
    # )
    # time_habitability_paramspace(
    #     "single_obliquity" + suff,
    #     here,
    #     obliquity_name,
    #     obliquity_unit,
    #     180,
    #     200,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    #     # vertical_lines=[
    #     #     (EARTH_MIN_OBLIQUITY, "dashed", "Earth Min"),
    #     #     (EARTH_MAX_OBLIQUITY, "dashdot", "Earth Max"),
    #     # ],
    # )

    # area_habitability_paramspace(
    #     "single_landfrac" + suff,
    #     here,
    #     landfrac_name,
    #     landfrac_unit,
    #     180,
    #     182,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    #     # vertical_lines=[
    #     #     (EARTH_MIN_OBLIQUITY, "dashed", "Earth Min"),
    #     #     (EARTH_MAX_OBLIQUITY, "dashdot", "Earth Max"),
    #     # ],
    # )
    # time_habitability_paramspace(
    #     "single_landfrac" + suff,
    #     here,
    #     landfrac_name,
    #     landfrac_unit,
    #     180,
    #     182,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    #     # vertical_lines=[
    #     #     (EARTH_MIN_OBLIQUITY, "dashed", "Earth Min"),
    #     #     (EARTH_MAX_OBLIQUITY, "dashdot", "Earth Max"),
    #     # ],
    # )
    # area_habitability_paramspace(
    #     "single_moonsemimajoraxis",
    #     here,
    #     amoon_name,
    #     a_unit,
    #     180,
    #     182,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    # )
    # time_habitability_paramspace(
    #     "single_moonsemimajoraxis",
    #     here,
    #     amoon_name,
    #     a_unit,
    #     180,
    #     200,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    # )
    # area_habitability_paramspace(
    #     "single_mooneccentricity",
    #     here,
    #     emoon_name,
    #     e_unit,
    #     180,
    #     182,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    #     # vertical_lines=[
    #     #     (EARTH_CURRENT_ECCENTRICITY, "dashed", "Earth Current"),
    #     #     (EARTH_MAX_ECCENTRICITY, "dashdot", "Earth Max"),
    #     # ],
    # )
    # time_habitability_paramspace(
    #     "single_mooneccentricity",
    #     here,
    #     emoon_name,
    #     e_unit,
    #     180,
    #     200,
    #     H=HumanCompatible,
    #     H_label="(HC)",
    # )

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
    #     "dual_gassemimajoraxis_gaseccentricity",
    #     os.path.curdir,
    #     "a$_{gas}$",
    #     "e$_{gas}$",
    #     a_unit,
    #     e_unit,
    #     yearstart=190,
    #     yearlength=10,
    #     H=HumanCompatible,
    # )
    habitability_paramspace(
        "dual_moonsemimajoraxis_mooneccentricity",
        here,
        amoon_name,
        emoon_name,
        a_unit,
        e_unit,
        180,
        3,
        HumanCompatible,
        "(HC)",
    )
