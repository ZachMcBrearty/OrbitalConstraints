from typing import Optional, Literal
import os
import multiprocessing as mp
from time import time

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from Constants import *
from ClimateModel import run_climate_model
from filemanagement import (
    load_config,
    write_to_file,
    read_dual_folder,
    read_single_folder,
)
from plotting import *


def convergence_test(
    temps: NDArray,
    rtol: float = 0.001,
    atol: float = 0,
    yearavg: float = 1,
    dt: float = 1,
) -> tuple[float, float]:
    """returns: how long the data set took to converge in years, -1 if never"""
    # data should be in year-long chunks,
    # find the length of extra data which doesnt fit into a year,
    # and then skip from the start of the dataset
    year_len = int(365 / dt)
    # -> i.e. each timestep is dt, so a year is 365 / dt datapoints long
    spacedim = temps.shape[0]
    a = int(temps.size % (year_len * spacedim * yearavg) / spacedim)
    tq = temps[:, a:].reshape(spacedim, -1, int(year_len * yearavg))
    tqa = np.average(tq, axis=2)  # average over time
    tqaa = np.average(
        tqa, axis=0, weights=np.cos(np.linspace(-np.pi / 2, np.pi / 2, temps.shape[0]))
    )  # average over latitude, weighted by area

    i = 0
    imax = len(tqaa) - 2
    while not np.isclose(tqaa[i], tqaa[i + 1], rtol=rtol, atol=atol) and i != imax:
        i += 1
    if i == imax or (tqaa[i] < 0):
        return -1.0, -1.0
    else:
        return i * yearavg, tqaa[i]


def _do_single_test(
    conf_name: str,
    val_sec: str,
    val_name: str,
    val: float,
    rounding_dp=3,
    verbose=True,
):
    conf = load_config(conf_name)
    if not conf.has_option(val_sec, val_name):
        raise ValueError(
            f"Supplied config file either has no section {val_sec} or option {val_name}"
        )
    if verbose:
        print(f"Running {val_name}={round(val, rounding_dp)}")
    conf.set(val_sec, val_name, str(val))
    degs, temps, times = run_climate_model(conf)
    filename = f"./single_{val_name}/single_{val_name}_{round(val,rounding_dp)}.npz"
    write_to_file(times, temps, degs, filename)


def parallel_convergence_test(
    val_sec: str,
    val_name: str,
    verbose=True,
):
    def t(conf_name, val_range, rounding_dp=3):
        if not os.path.exists(f"./single_{val_name}/"):
            os.mkdir(f"./single_{val_name}/")
        t0 = 0
        if verbose:
            t0 = time()
            print(f"Starting {val_name}")
        with mp.Pool() as p:
            p.starmap(
                _do_single_test,
                [
                    (conf_name, val_sec, val_name, val, rounding_dp, verbose)
                    for val in val_range
                ],
            )
        if verbose:
            tf = time()
            dt = tf - t0
            if dt > 3600:
                print(
                    f"Finished {val_name} in {dt//3600} hours {(dt %3600)//60} min {(dt %3600)%60} secs"
                )
            elif dt > 60:
                print(f"Finished {val_name} in {dt // 60} min {dt%60} secs")
            else:
                print(f"Finished {val_name} in {dt} seconds")

    return t


def gen_convergence_test(
    val_section: str,
    val_name: str,
    verbose=False,
    plot=False,
    save=True,
    val_unit: Optional[str] = None,
):
    # if val_unit is None:
    #     val_unit = ""
    # else:
    #     val_unit = ", " + val_unit

    def t(
        conf: CONF_PARSER_TYPE,
        val_range: list | NDArray,
        rtol=0.0001,
    ) -> tuple[list | NDArray, list[float], list[float]]:
        if not conf.has_option(val_section, val_name):
            raise ValueError(
                f"Supplied config file has no section {val_section} or option {val_name}"
            )
        tests = []
        convtemps = []
        # val_range = np.arange(val_min, val_max, val_step)
        for val in val_range:
            if verbose:
                print(f"Running {val_name}={val}")
            conf.set(val_section, val_name, str(val))
            degs, temps, times = run_climate_model(conf)
            t, temp = convergence_test(
                temps, rtol, yearavg=1, dt=conf.getfloat("PDE", "timestep")
            )
            tests.append(t)
            convtemps.append(temp)
            if save:
                if not os.path.exists(f"./single_{val_name}/"):
                    os.mkdir(f"./single_{val_name}/")
                filename = f"./single_{val_name}/single_{val_name}_{round(val,5)}.npz"
                write_to_file(
                    times,
                    temps,
                    degs,
                    filename,
                )

        if plot:
            convergence_plot_single(
                np.ndarray(tests, dtype=float),
                np.ndarray(convtemps, dtype=float),
                val_name,
                val_range,
                val_unit,
            )

        return val_range, tests, convtemps

    return t


test_spacedim_convergence = gen_convergence_test(
    "PDE", "spacedim", True, False, True, spacedim_unit
)
test_timestep_convergence = gen_convergence_test(
    "PDE", "timestep", True, False, True, timestep_unit
)
test_temp_convergence = gen_convergence_test(
    "PDE", "starttemp", True, False, True, temp_unit
)

test_omega_convergence = parallel_convergence_test("PLANET", "omega", True)
test_delta_convergence = parallel_convergence_test("PLANET", "obliquity", True)

test_a_convergence = parallel_convergence_test("ORBIT", "gassemimajoraxis", True)
test_e_convergence = parallel_convergence_test("ORBIT", "gaseccentricity", True)
test_moon_a_convergence = parallel_convergence_test("ORBIT", "moonsemimajoraxis", True)
test_moon_e_convergence = parallel_convergence_test("ORBIT", "mooneccentricity", True)

test_moonrad_convergence = parallel_convergence_test("ORBIT", "moonradius", True)
test_moondensity_convergence = parallel_convergence_test("ORBIT", "moondensity", True)
test_gasmass_convergence = parallel_convergence_test("ORBIT", "gasgiantmass", True)

test_ocean_fraction_convergence = parallel_convergence_test("PLANET", "landfrac")


def _do_dual_test(
    conf_name: str,
    val_1_sec,
    val_1_name,
    val_1,
    val_2_sec,
    val_2_name,
    val_2,
    rounding_dp=3,
    verbose=True,
    overwrite=False,
) -> None:
    conf = load_config(conf_name)
    if not conf.has_option(val_1_sec, val_1_name):
        raise ValueError(
            f"Supplied config file has no section {val_1_sec} or option {val_1_name}"
        )
    if not conf.has_option(val_2_sec, val_2_name):
        raise ValueError(
            f"Supplied config file has no section {val_2_sec} or option {val_2_name}"
        )

    filename = f"./dual_{val_1_name}_{val_2_name}/dual_{val_1_name}_{round(val_1,rounding_dp)}_{val_2_name}_{round(val_2, rounding_dp)}.npz"
    if not overwrite and os.path.exists(filename):
        if verbose:
            print(
                f"Skipping {val_1_name}={round(val_1, rounding_dp)}, {val_2_name}={round(val_2, rounding_dp)}"
            )
        return
    elif verbose:
        print(
            f"Running {val_1_name}={round(val_1, rounding_dp)}, {val_2_name}={round(val_2, rounding_dp)}"
        )
    conf.set(val_1_sec, val_1_name, str(val_1))
    conf.set(val_2_sec, val_2_name, str(val_2))
    degs, temps, times = run_climate_model(conf)

    write_to_file(times, temps, degs, filename)


def parallel_gen_paramspace(
    val_1_sec: str,
    val_1_name: str,
    val_2_sec: str,
    val_2_name: str,
    verbose=False,
):
    def t(
        conf_name: str,
        val_1_range: list | NDArray,
        val_2_range: list | NDArray,
        rounding_dp=3,
        overwrite=False,
    ) -> None:
        if not os.path.exists(f"./dual_{val_1_name}_{val_2_name}/"):
            os.mkdir(f"./dual_{val_1_name}_{val_2_name}/")
        if verbose:
            print(f"Starting {val_1_name} {val_2_name}")
        pairs = []
        for val_1 in val_1_range:
            for val_2 in val_2_range:
                pairs.append(
                    (
                        conf_name,
                        val_1_sec,
                        val_1_name,
                        val_1,
                        val_2_sec,
                        val_2_name,
                        val_2,
                        rounding_dp,
                        verbose,
                        overwrite,
                    )
                )
        t0 = time()
        with mp.Pool() as p:
            p.starmap(
                _do_dual_test,
                pairs,
            )
        if verbose:
            tf = time()
            dt = tf - t0
            if dt > 3600:
                print(
                    f"Finished {val_1_name} {val_2_name} in {dt//3600} hours {(dt %3600)//60} min {(dt %3600)%60} secs"
                )
            elif dt > 60:
                print(
                    f"Finished {val_1_name} {val_2_name} in {dt // 60} min {dt%60} secs"
                )
            else:
                print(f"Finished {val_1_name} {val_2_name} in {dt} seconds")
            # print(f"Finished {val_1_name} {val_2_name}")

    return t


def gen_paramspace(
    val_sec_1: str,
    val_name_1: str,
    val_sec_2: str,
    val_name_2: str,
    verbose=False,
    plot=False,
    val_unit_1: Optional[str] = None,
    val_unit_2: Optional[str] = None,
    save=False,
):
    def t(
        conf: CONF_PARSER_TYPE,
        val_1_range: list | NDArray,
        val_2_range: list | NDArray,
        rtol=0.0001,
        rounding_dp=3,
    ) -> tuple[list | NDArray, list | NDArray, NDArray, NDArray]:
        if not conf.has_option(val_sec_1, val_name_1):
            raise ValueError(
                f"Supplied config file has no section {val_sec_1} or option {val_name_1}"
            )
        if not conf.has_option(val_sec_2, val_name_2):
            raise ValueError(
                f"Supplied config file has no section {val_sec_2} or option {val_name_2}"
            )
        # val_1_range = np.arange(val_min_1, val_max_1, val_step_1)
        # val_2_range = np.arange(val_min_2, val_max_2, val_step_2)
        tests = np.zeros((len(val_1_range), len(val_2_range)))
        convtemps = np.zeros_like(tests)
        for i, val_1 in enumerate(val_1_range):
            for j, val_2 in enumerate(val_2_range):
                if verbose:
                    print(
                        f"Running {val_name_1}[{i}]={val_1}, {val_name_2}[{j}]={val_2}"
                    )
                conf.set(val_sec_1, val_name_1, str(val_1))
                conf.set(val_sec_2, val_name_2, str(val_2))
                degs, temps, times = run_climate_model(conf)
                t, temp = convergence_test(
                    temps, rtol, yearavg=1, dt=conf.getfloat("PDE", "timestep")
                )
                tests[i][j] = t
                convtemps[i][j] = temp
                if save:
                    if not os.path.exists(f"./dual_{val_name_1}_{val_name_2}/"):
                        os.mkdir(f"./dual_{val_name_1}_{val_name_2}/")
                    filename = f"./dual_{val_name_1}_{val_name_2}/dual_{val_name_1}_{round(val_1,rounding_dp)}_{val_name_2}_{round(val_2, rounding_dp)}.npz"
                    write_to_file(times, temps, degs, filename)
        if plot:
            convergence_plot_dual(
                tests,
                convtemps,
                val_name_1,
                val_1_range,
                val_name_2,
                val_2_range,
                val_unit_1,
                val_unit_2,
            )

        return val_1_range, val_2_range, tests, convtemps

    return t


dual_a_e_convergence = gen_paramspace(
    "ORBIT",
    "a",
    "ORBIT",
    "e",
    verbose=True,
    plot=True,
    val_unit_1=a_unit,
    val_unit_2=e_unit,
    save=True,
)
dual_a_e_convergence_parallel = parallel_gen_paramspace(
    "ORBIT", "gassemimajoraxis", "ORBIT", "gaseccentricity", verbose=True
)
dual_a_delta_convergence = gen_paramspace(
    "ORBIT",
    "a",
    "PLANET",
    "obliquity",
    verbose=True,
    plot=False,
    val_unit_1=a_unit,
    val_unit_2=obliquity_unit,
    save=True,
)

dual_a_omega_convergence = gen_paramspace(
    "ORBIT",
    "a",
    "PLANET",
    "omega",
    True,
    False,
    a_unit,
    omega_unit,
    True,
)

dual_a_temp_convergence = gen_paramspace(
    "ORBIT",
    "a",
    "PDE",
    "starttemp",
    True,
    False,
    a_unit,
    temp_unit,
    True,
)

dual_e_delta_convergence = gen_paramspace(
    "ORBIT", "e", "PLANET", "obliquity", True, False, e_unit, obliquity_unit, True
)

dual_e_omega_convergence = gen_paramspace(
    "ORBIT", "e", "PLANET", "omega", True, False, e_unit, omega_unit, True
)

dual_e_starttemp_convergence = gen_paramspace(
    "ORBIT", "e", "PDE", "starttemp", True, False, e_unit, temp_unit, True
)

dual_delta_omega_convergence = gen_paramspace(
    "PLANET",
    "obliquity",
    "PLANET",
    "omega",
    True,
    False,
    obliquity_unit,
    omega_unit,
    True,
)

dual_delta_starttemp_convergence = gen_paramspace(
    "PLANET",
    "obliquity",
    "PDE",
    "starttemp",
    True,
    False,
    obliquity_unit,
    temp_unit,
    True,
)

dual_omega_starttemp_convergence = gen_paramspace(
    "PLANET", "omega", "PDE", "starttemp", True, False, omega_unit, temp_unit, True
)

dual_moon_a_e_convergence = parallel_gen_paramspace(
    "ORBIT", "moonsemimajoraxis", "ORBIT", "mooneccentricity", True
)

dual_e_landfrac_convergence = parallel_gen_paramspace(
    "ORBIT", "gaseccentricity", "PLANET", "landfrac", True
)

dual_delta_landfrac_convergence = parallel_gen_paramspace(
    "PLANET", "obliquity", "PLANET", "landfrac", True
)


def process_data_single(
    foldername: str, folderpath: str, yearavg: int = 1, rtol: float = 1e-3
) -> tuple[NDArray, NDArray, NDArray]:
    """returns: val_range, tests, convtemps"""
    val_range = []
    tests = []
    convtemps = []
    data = read_single_folder(foldername, folderpath)
    for i, datum in enumerate(data):
        val_1, (times, temps, degs) = datum
        dt = (times[1] - times[0]) * 365
        t, temp = convergence_test(temps, rtol, yearavg=yearavg, dt=dt)
        if (q := float(val_1)) not in val_range:
            val_range.append(q)
        tests.append(t)
        convtemps.append(temp)
    return np.array(val_range), np.array(tests), np.array(convtemps)


def reprocess_single_param(
    foldername: str,
    folderpath: str,
    val_name: str,
    val_unit: Optional[str] = None,
    rtol: float = 0.0001,
    yearavg=1,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    val_range, tests, convtemps = process_data_single(
        foldername, folderpath, yearavg, rtol
    )
    convergence_plot_single(
        tests,
        convtemps,
        val_name,
        val_range,
        val_unit,
        x_axis_scale=x_axis_scale,
        y_axis_scale=y_axis_scale,
    )


def reprocess_ecc_fit(
    foldername: str,
    folderpath: str,
    val_name: str = egas_name,
    val_unit: Optional[str] = e_unit,
    rtol: float = 0.0001,
    yearavg=1,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
    plus_b=False,
):
    val_range, tests, convtemps = process_data_single(
        foldername, folderpath, yearavg, rtol
    )
    if plus_b:
        func = lambda x, a, b: (a + b / np.sqrt(1 - x**2)) ** (1 / 4)
        func_label = r"$(p + q (1-e^2)^{-1/2})^{1/4}$"
    else:
        func = lambda x, a: a * (1 - x**2) ** (-1 / 8)
        func_label = r"$p (1-e^2)^{-1/8}$"
    single_variable_single_fit_plot(
        tests,
        convtemps,
        val_name,
        val_range,
        func,
        (0, -1),
        func_label,
        val_unit,
        x_axis_scale,
        y_axis_scale,
    )


def reprocess_semimajor_fit(
    foldername: str,
    folderpath: str,
    val_name: str = agas_name,
    val_unit: Optional[str] = a_unit,
    rtol: float = 0.0001,
    yearavg=1,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
    split=51,
):
    val_range, tests, convtemps = process_data_single(
        foldername, folderpath, yearavg, rtol
    )
    if split == -1:
        single_variable_single_fit_plot(
            tests,
            convtemps,
            val_name,
            val_range,
            lambda x, a, b: (a + b / x**2) ** (1 / 4),
            (0, -1),
            r"$T = (p + q a^{-2})^{1/4}$",
            val_unit,
            x_axis_scale,
            y_axis_scale,
        )
    else:
        single_variable_N_fits_plot(
            tests,
            convtemps,
            val_name,
            val_range,
            [lambda x, a, b: a * x ** (-b), lambda x, a, b: a * x ** (-b)],
            [(0, split), (split, -1)],
            [r"$p_1 a^{q_1}$", r"$p_2 a^{q_2}$"],
            val_unit,
            x_axis_scale,
            y_axis_scale,
        )


def reprocess_moon_ecc_fit(
    foldername: str,
    folderpath: str,
    val_name: str = emoon_name,
    val_unit: Optional[str] = e_unit,
    rtol: float = 0.0001,
    yearavg=1,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    val_range, tests, convtemps = process_data_single(
        foldername, folderpath, yearavg, rtol
    )
    single_variable_single_fit_plot(
        tests,
        convtemps,
        val_name,
        val_range,
        lambda x, a, b: (b * x**2 + a) ** (1 / 4),
        (0, -1),
        r"$(p + q x^2)^{-1/4}$",
        val_unit,
        x_axis_scale,
        y_axis_scale,
    )


def reprocess_moon_semimajor_fit(
    foldername: str,
    folderpath: str,
    val_name: str = amoon_name,
    val_unit: Optional[str] = a_unit,
    rtol: float = 0.0001,
    yearavg=1,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    val_range, tests, convtemps = process_data_single(
        foldername, folderpath, yearavg, rtol
    )
    single_variable_single_fit_plot(
        tests,
        convtemps,
        val_name,
        val_range,
        lambda x, a, b: (a + b * x ** (-15 / 2)) ** (1 / 4),
        (0, -1),
        r"(p + q x^{-15/2})^{1/4}",
        val_unit,
        x_axis_scale,
        y_axis_scale,
    )


def reprocess_single_param_compare(
    foldername_1: str,
    folderpath_1: str,
    foldername_2: str,
    folderpath_2: str,
    val_name_1: str,
    val_name_2: str,
    val_unit: Optional[str] = None,
    rtol: float = 0.0001,
    yearavg=1,
):
    val_range_1, tests_1, convtemps_1 = process_data_single(
        foldername_1, folderpath_1, yearavg, rtol
    )
    val_range_2, tests_2, convtemps_2 = process_data_single(
        foldername_2, folderpath_2, yearavg, rtol
    )

    convergence_plot_single_compare(
        tests_1,
        convtemps_1,
        val_name_1,
        val_range_1,
        tests_2,
        convtemps_2,
        val_name_2,
        val_range_2,
        val_unit,
    )


def process_data_double(
    foldername: str, folderpath: str, yearavg: int = 1, rtol: float = 1e-3
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """returns: val_range_1, val_range_2, tests, convtemps"""
    val_range_1 = []
    val_range_2 = []
    tests = []
    convtemps = []
    data = read_dual_folder(foldername, folderpath)
    for i, datum in enumerate(data):
        val_1, val_2, (times, temps, degs) = datum
        dt = (times[1] - times[0]) * 365
        t, temp = convergence_test(temps, rtol, yearavg=1, dt=dt)
        if (q := float(val_1)) not in val_range_1:
            val_range_1.append(q)
        if (q := float(val_2)) not in val_range_2:
            val_range_2.append(q)
        tests.append(t)
        convtemps.append(temp)
    xl = len(val_range_1)
    yl = len(val_range_2)
    tests = np.array(tests).reshape(xl, yl)
    convtemps = np.array(convtemps).reshape(xl, yl)
    return np.array(val_range_1), np.array(val_range_2), tests, convtemps


def reprocess_paramspace(
    foldername: str,
    folderpath: str,
    val_name_1: str,
    val_name_2: str,
    val_unit_1: Optional[str] = None,
    val_unit_2: Optional[str] = None,
    rtol: float = 0.0001,
    yearavg: int = 1,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    val_range_1, val_range_2, tests, convtemps = process_data_double(
        foldername, folderpath, yearavg, rtol
    )
    convergence_plot_dual(
        tests,
        convtemps,
        val_name_1,
        val_range_1,
        val_name_2,
        val_range_2,
        val_unit_1,
        val_unit_2,
        x_axis_scale,
        y_axis_scale,
    )


def reprocess_dual_compare(
    foldernames: tuple[str, str],
    folderpaths: tuple[str, str],
    val_name_a: str,
    val_name_b: str,
    val_unit_a: Optional[str] = None,
    val_unit_b: Optional[str] = None,
    rtol: float = 0.0001,
    yearavg: int = 1,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    val_range_1_a, val_range_2_a, tests_a, convtemps_a = process_data_double(
        foldernames[0], folderpaths[0], yearavg, rtol
    )
    val_range_1_b, val_range_2_b, tests_b, convtemps_b = process_data_double(
        foldernames[1], folderpaths[1], yearavg, rtol
    )
    convergence_plot_dual_compare(
        tests_a,
        convtemps_a,
        tests_b,
        convtemps_b,
        val_name_a,
        val_range_1_a,
        val_range_2_a,
        val_name_b,
        val_range_1_b,
        val_range_2_b,
        val_unit_a,
        val_unit_b,
        x_axis_scale,
        y_axis_scale,
    )


if __name__ == "__main__":
    here = os.path.curdir
    # here = "D:/v2/"
    conf = "config.ini"
    conf_moon = "config_moon.ini"
    # th = "single_obliquity"
    # val_range_1, tests_1, convtemps_1 = process_data_single(f"{th}", here, rtol=1e-3)
    # val_range_2, tests_2, convtemps_2 = process_data_single(
    #     f"{th}_TH_0.003_0.006", here, rtol=1e-3
    # )
    # val_range_3, tests_3, convtemps_3 = process_data_single(
    #     f"{th}_TH_0.003_0.01", here, rtol=1e-3
    # )

    # plot_three_on_one_graph(
    #     (tests_1, tests_2, tests_3),
    #     (convtemps_1, convtemps_2, convtemps_3),
    #     (val_range_1, val_range_2, val_range_3),
    #     obliquity_name,
    #     obliquity_unit,
    #     (
    #         "No Tidal heating",
    #         r"$a_{moon} = 0.003, e_{moon}=0.006$",
    #         r"$a_{moon} = 0.003, e_{moon}=0.01$",
    #     ),
    # )
    # dual_a_e_convergence_parallel(
    #     conf, np.linspace(0.5, 2, 31), np.linspace(0, 0.9, 31), 5
    # )
    # reprocess_paramspace(
    #     "dual_gassemimajoraxis_gaseccentricity",
    #     here,
    #     agas_name,
    #     egas_name,
    #     a_unit,
    #     e_unit,
    #     rtol=1e-2,
    # )

    # test_omega_convergence(conf, np.linspace(0.5, 3, 41), 5)
    # test_omega_convergence(conf, [0.25], 5)
    # reprocess_single_param("single_omega", here, omega_name, omega_unit, rtol=1e-5)

    # test_a_convergence(conf, np.linspace(4, 6, 41), 5)
    # test_e_convergence(conf, np.linspace(0, 0.9, 51), 5)
    # test_delta_convergence(conf, np.linspace(0, 90, 101), 5)
    # test_delta_convergence(conf, np.linspace(0, 180, 21), 5)
    # reprocess_semimajor_fit(
    #     "single_gassemimajoraxis",
    #     here,
    #     val_name=agas_name,
    #     rtol=1e-4,
    #     split=-1,
    # )
    # reprocess_ecc_fit(
    #     "single_gaseccentricity",
    #     here,
    #     val_name=egas_name,
    #     rtol=1e-5,
    #     plus_b=True,
    # )
    # reprocess_single_param(
    #     "single_obliquity", here, obliquity_name, obliquity_unit, 1e-5
    # )

    # test_ocean_fraction_convergence(
    #     conf, [f"uniform:{q:.3f}" for q in np.linspace(0, 1, 51)], -1
    # )

    # reprocess_single_param(
    #     "single_landfractype", here, "Uniform ocean fraction", None, 1e-5
    # )

    # dual_e_landfrac_convergence(
    #     conf,
    #     np.linspace(0, 0.9, 21),
    #     np.linspace(0, 1, 51),
    #     5,
    # )
    reprocess_paramspace(
        "dual_gaseccentricity_landfrac",
        here,
        eplt_name,
        "Uniform ocean fraction",
        e_unit,
        None,
        1e-3,
    )
    # dual_delta_landfrac_convergence(
    #     conf,
    #     np.linspace(0, 90, 21),
    #     np.linspace(0, 1, 51),
    #     5,
    # )

    # reprocess_paramspace(
    #     "dual_obliquity_landfrac",
    #     here,
    #     obliquity_name,
    #     "Uniform ocean fraction",
    #     obliquity_unit,
    #     None,
    #     1e-3,
    # )

    # test_moon_a_convergence(conf_moon, np.linspace(0.001, 0.01, 51), 5)
    # test_moon_e_convergence(conf_moon, np.linspace(0, 0.1, 51), 5)
