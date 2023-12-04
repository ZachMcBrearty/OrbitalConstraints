from typing import Optional, Literal
import os
import multiprocessing as mp

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from Constants import *
from ClimateModel import climate_model_in_lat
from filemanagement import (
    load_config,
    write_to_file,
    read_dual_folder,
    read_single_folder,
)
from plotting import (
    convergence_plot_dual_with_fits,
    convergence_plot_single,
    convergence_plot_dual,
    convergence_plot_single_compare,
    ecc_fit_plot,
    semimajor_fit_plot,
)


def convergence_test(
    temps: NDArray,
    rtol: float = 0.001,
    atol: float = 0,
    year_avg: int = 1,
    dt: float = 1,
) -> tuple[int, float]:
    """returns: how long the data set took to converge in years, -1 if never"""
    # data should be in year-long chunks,
    # find the length of extra data which doesnt fit into a year,
    # and then skip from the start of the dataset
    year_len = int(365 / dt)
    # -> i.e. each timestep is dt, so a year is 365 / dt datapoints long
    spacedim = temps.shape[0]
    a = int(temps.size % (year_len * spacedim * year_avg) / spacedim)
    tq = temps[:, a:].reshape(spacedim, -1, year_len * year_avg)
    tqa = np.average(tq, axis=2)  # average over time
    tqaa = np.average(
        tqa, axis=0, weights=np.cos(np.linspace(-np.pi / 2, np.pi / 2, temps.shape[0]))
    )  # average over latitude, weighted by area

    i = 0
    imax = len(tqaa) - 2
    while not np.isclose(tqaa[i], tqaa[i + 1], rtol=rtol, atol=atol) and i != imax:
        i += 1
    if i == imax:
        return -1, 0
    else:
        return i * year_avg, tqaa[i]


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
            degs, temps, times = climate_model_in_lat(conf)
            t, temp = convergence_test(
                temps, rtol, year_avg=1, dt=conf.getfloat("PDE", "timestep")
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
                tests,
                convtemps,
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

test_omega_convergence = gen_convergence_test(
    "PLANET", "omega", True, False, True, omega_unit
)
test_delta_convergence = gen_convergence_test(
    "PLANET", "obliquity", True, False, True, obliquity_unit
)

test_a_convergence = gen_convergence_test(
    "ORBIT", "gassemimajoraxis", True, False, True, a_unit
)
test_e_convergence = gen_convergence_test(
    "ORBIT", "gaseccentricity", True, False, True, e_unit
)
test_moon_a_convergence = gen_convergence_test(
    "ORBIT", "moonsemimajoraxis", True, False, True, a_unit
)
test_moon_e_convergence = gen_convergence_test(
    "ORBIT", "mooneccentricity", True, False, True, a_unit
)


def _do_test(
    conf_name: str,
    val_1_sec,
    val_1_name,
    val_1,
    val_2_sec,
    val_2_name,
    val_2,
    rounding_dp=3,
    verbose=True,
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
    if verbose:
        print(f"Running {val_1_name}={val_1}, {val_2_name}={val_2}")
    conf.set(val_1_sec, val_1_name, str(val_1))
    conf.set(val_2_sec, val_2_name, str(val_2))
    degs, temps, times = climate_model_in_lat(conf)
    filename = f"./dual_{val_1_name}_{val_2_name}/dual_{val_1_name}_{round(val_1,rounding_dp)}_{val_2_name}_{round(val_2, rounding_dp)}.npz"
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
                    )
                )
        with mp.Pool() as p:
            p.starmap(
                _do_test,
                pairs,
            )
        if verbose:
            print(f"Finished {val_1_name} {val_2_name}")

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
                degs, temps, times = climate_model_in_lat(conf)
                t, temp = convergence_test(
                    temps, rtol, year_avg=1, dt=conf.getfloat("PDE", "timestep")
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
                rtol,
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

dual_moon_a_e_convergence = gen_paramspace(
    "ORBIT",
    "moonsemimajoraxis",
    "ORBIT",
    "mooneccentricity",
    True,
    False,
    a_unit,
    e_unit,
    True,
)


def reprocess_single_param(
    foldername: str,
    folderpath: str,
    val_1_name: str,
    val_1_unit: Optional[str] = None,
    rtol: float = 0.0001,
    x_axis_scale: Literal["linear", "log"] = "linear",
):
    val_1_range = []
    tests = []
    convtemps = []
    data = read_single_folder(foldername, folderpath)
    for i, datum in enumerate(data):
        val_1, (times, temps, degs) = datum
        dt = (times[1] - times[0]) * 365
        t, temp = convergence_test(temps, rtol, year_avg=1, dt=dt)
        if (q := float(val_1)) not in val_1_range:
            val_1_range.append(q)
        tests.append(t)
        convtemps.append(temp)
    convergence_plot_single(
        tests, convtemps, val_1_name, val_1_range, val_1_unit, x_axis_scale=x_axis_scale
    )


def reprocess_ecc_fit(
    foldername: str,
    folderpath: str,
    val_1_name: str,
    val_1_unit: Optional[str] = None,
    rtol: float = 0.0001,
    x_axis_scale: Literal["linear", "log"] = "linear",
):
    val_1_range = []
    tests = []
    convtemps = []
    data = read_single_folder(foldername, folderpath)
    for i, datum in enumerate(data):
        val_1, (times, temps, degs) = datum
        dt = (times[1] - times[0]) * 365
        t, temp = convergence_test(temps, rtol, year_avg=1, dt=dt)
        if (q := float(val_1)) not in val_1_range:
            val_1_range.append(q)
        tests.append(t)
        convtemps.append(temp)
    ecc_fit_plot(
        tests,
        convtemps,
        val_1_name,
        val_1_range,
        val_1_unit,
    )


def reprocess_semimajor_fit(
    foldername: str,
    folderpath: str,
    val_1_name: str,
    val_1_unit: Optional[str] = None,
    rtol: float = 0.0001,
    x_axis_scale: Literal["linear", "log"] = "linear",
):
    val_1_range = []
    tests = []
    convtemps = []
    data = read_single_folder(foldername, folderpath)
    for i, datum in enumerate(data):
        val_1, (times, temps, degs) = datum
        dt = (times[1] - times[0]) * 365
        t, temp = convergence_test(temps, rtol, year_avg=1, dt=dt)
        if (q := float(val_1)) not in val_1_range:
            val_1_range.append(q)
        tests.append(t)
        convtemps.append(temp)
    semimajor_fit_plot(
        tests,
        convtemps,
        val_1_name,
        val_1_range,
        val_1_unit,
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
):
    val_range_1 = []
    tests_1 = []
    convtemps_1 = []
    data_1 = read_single_folder(foldername_1, folderpath_1)
    for i, datum in enumerate(data_1):
        val_1, (times, temps, degs) = datum
        dt = (times[1] - times[0]) * 365
        t, temp = convergence_test(temps, rtol, year_avg=1, dt=dt)
        if (q := float(val_1)) not in val_range_1:
            val_range_1.append(q)
        tests_1.append(t)
        convtemps_1.append(temp)

    val_range_2 = []
    tests_2 = []
    convtemps_2 = []
    data_2 = read_single_folder(foldername_2, folderpath_2)
    for i, datum in enumerate(data_2):
        val_2, (times, temps, degs) = datum
        dt = (times[1] - times[0]) * 365
        t, temp = convergence_test(temps, rtol, year_avg=1, dt=dt)
        if (q := float(val_2)) not in val_range_2:
            val_range_2.append(q)
        tests_2.append(t)
        convtemps_2.append(temp)

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


def reprocess_paramspace(
    foldername: str,
    folderpath: str,
    val_1_name: str,
    val_2_name: str,
    val_1_unit: Optional[str] = None,
    val_2_unit: Optional[str] = None,
    rtol: float = 0.0001,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    val_1_range = []
    val_2_range = []
    tests = []
    convtemps = []
    data = read_dual_folder(foldername, folderpath)
    for i, datum in enumerate(data):
        val_1, val_2, (times, temps, degs) = datum
        dt = (times[1] - times[0]) * 365
        t, temp = convergence_test(temps, rtol, year_avg=1, dt=dt)
        if (q := float(val_1)) not in val_1_range:
            val_1_range.append(q)
        if (q := float(val_2)) not in val_2_range:
            val_2_range.append(q)
        tests.append(t)
        convtemps.append(temp)
    xl = len(val_1_range)
    yl = len(val_2_range)
    tests = np.array(tests).reshape(xl, yl)
    convtemps = np.array(convtemps).reshape(xl, yl)
    convergence_plot_dual_with_fits(
        # convergence_plot_dual(
        tests,
        convtemps,
        val_1_name,
        np.array(val_1_range),
        val_2_name,
        np.array(val_2_range),
        val_1_unit,
        val_2_unit,
        x_axis_scale,
        y_axis_scale,
    )


def reset_conf(conf):
    conf.set("FILEMANAGEMENT", "save", "False")
    conf.set("FILEMANAGEMENT", "plot", "False")

    conf.set("PDE", "spacedim", "60")  #
    conf.set("PDE", "time", "200")
    conf.set("PDE", "timestep", "1")  #
    conf.set("PDE", "starttemp", "350")  #

    conf.set("PLANET", "omega", "1")  #
    conf.set("PLANET", "landfractype", "uniform:0.7")
    conf.set("PLANET", "obliquity", "23.5")  #

    conf.set("ORBIT", "a", "1")  #
    conf.set("ORBIT", "e", "0")  #
    return conf


if __name__ == "__main__":
    # print(test_a_convergence(conf, 0.5, 2, 0.1, rtol=0.0001))
    # conf = load_config("config.ini")
    # print(test_moon_a_convergence(conf, np.arange(0.001, 0.01, 0.001)))
    # conf = load_config("config.ini")
    # print(test_moon_e_convergence(conf, np.logspace(-3, -1, 20)))
    # print(test_a_convergence(conf, 2, 10, 0.5, rtol=0.0001))
    # print(test_e_convergence(conf, 0, 0.91, 0.1, rtol=0.0001))
    # reset_conf(conf)
    # print(test_delta_convergence(conf, 0, 181, 10, rtol=0.0001))
    # reset_conf(conf)
    # print(test_omega_convergence(conf, 2.4, 2.45, 0.005, rtol=0.0001))
    # reset_conf(conf)
    # print(test_temp_convergence(conf, 100, 501, 50, rtol=0.0001))
    # reset_conf(conf)
    # print(test_spacedim_convergence(conf, 30, 180, 15, rtol=0.0001))
    # conf.set("PDE", "time", "300")
    # print(test_timestep_convergence(conf, 0.25, 3.1, 0.25, rtol=0.0001))

    # reset_conf(conf)
    # conf = load_config("config.ini")
    # dual_a_e_convergence_parallel(
    #     "config.ini", np.linspace(0.5, 2, 20), np.linspace(0, 0.9, 20), 3
    # )
    # print(dual_a_e_convergence(conf, 0.5, 2.05, 0.1, 0, 0.91, 0.1, 0.001))
    # reset_conf(conf)
    # print(dual_a_delta_convergence(conf, 0.5, 2.05, 0.1, 0, 91, 10, 0.001))
    # reset_conf(conf)
    # print(dual_a_omega_convergence(conf, 0.5, 2.05, 0.1, 0.25, 3.1, 0.25, 0.001))
    # reset_conf(conf)
    # print(dual_a_temp_convergence(conf, 0.5, 2.05, 0.1, 150, 500, 50, 0.001))
    # reset_conf(conf)

    # print(dual_e_delta_convergence(conf, 0, 0.91, 0.1, 0, 91, 10, 0.001))
    # reset_conf(conf)
    # print(dual_e_omega_convergence(conf, 0, 0.91, 0.1, 0.25, 3.1, 0.25, 0.001))
    # reset_conf(conf)
    # print(dual_e_starttemp_convergence(conf, 0, 0.91, 0.1, 150, 500, 50, 0.001))
    # reset_conf(conf)

    # print(dual_delta_omega_convergence(conf, 70, 91, 10, 0.25, 3.1, 0.25, 0.001))
    # reset_conf(conf)
    # print(dual_delta_starttemp_convergence(conf, 0, 91, 10, 150, 500, 50, 0.001))
    # reset_conf(conf)

    # print(dual_omega_starttemp_convergence(conf, 0.25, 3.1, 0.25, 150, 500, 50, 0.001))
    # reset_conf(conf)

    # conf = load_config("config.ini")
    # print(
    #     dual_moon_a_e_convergence(
    #         conf,
    #         np.linspace(0.001, 0.005, 20),
    #         np.logspace(-4, -1, 20),
    #         rounding_dp=10,
    #     )
    # )
    # reprocess_paramspace(
    #     "dual_moonsemimajoraxis_mooneccentricity",
    #     os.path.curdir,
    #     "a$_{moon}$",
    #     "e$_{moon}$",
    #     a_unit,
    #     e_unit,
    #     0.0001,
    #     "linear",
    #     "log",
    # )
    # reprocess_ecc_fit("single_e", os.path.curdir, "e", e_unit, 0.0001)
    # reprocess_semimajor_fit(
    #     "single_moonsemimajoraxis", os.path.curdir, "a$_{moon}$", a_unit, 0.0001
    # )
    # reprocess_semimajor_fit(
    #     "single_gassemimajoraxis", os.path.curdir, "a$_{gas}$", a_unit, 0.0001
    # )
    # reprocess_single_param(
    #     "single_mooneccentricity", os.path.curdir, "e$_{moon}$", e_unit, 0.0001, "log"
    # )
    # reprocess_single_param_compare(
    #     "single_a",
    #     os.path.curdir,
    #     "single_gassemimajoraxis",
    #     os.path.curdir,
    #     "a",
    #     "a$_{gas}$",
    #     "au",
    # )
    # reprocess_single_param(
    #     "single_gassemimajoraxis", os.path.curdir, "a gas", a_unit, 0.0001
    # )
    # reprocess_single_param("single_a", os.path.curdir, "a", a_unit, 0.0001)
    # plt.show()
    reprocess_paramspace(
        "dual_gassemimajoraxis_gaseccentricity",
        os.path.curdir,
        "a",
        "e",
        "au",
        None,
        0.005,
    )
    # reprocess_paramspace(
    #     "dual_a_obliquity",
    #     os.path.curdir,
    #     "a",
    #     "obliquity",
    #     "au",
    #     r"$^{\circ}$",
    #     0.0001,
    # )
    # reprocess_paramspace(
    #     "dual_a_omega", os.path.curdir, "a", "omega", "au", "days$^{-1}$", 0.0001
    # )
    # reprocess_paramspace(
    #     "dual_a_starttemp", os.path.curdir, "a", "starttemp", "au", "K", 0.0001
    # )

    # reprocess_paramspace(
    #     "dual_e_obliquity",
    #     os.path.curdir,
    #     "e",
    #     "obliquity",
    #     None,
    #     r"$^{\circ}$",
    #     0.0001,
    # )
    # reprocess_paramspace(
    #     "dual_e_omega", os.path.curdir, "e", "omega", None, "days$^{-1}$", 0.0001
    # )
    # reprocess_paramspace(
    #     "dual_e_starttemp", os.path.curdir, "e", "starttemp", None, "K", 0.0001
    # )

    # reprocess_paramspace(
    #     "dual_obliquity_omega",
    #     os.path.curdir,
    #     "obliquity",
    #     "omega",
    #     r"$^{\circ}$",
    #     "days$^{-1}$",
    #     0.0001,
    # )
    # reprocess_paramspace(
    #     "dual_obliquity_starttemp",
    #     os.path.curdir,
    #     "obliquity",
    #     "starttemp",
    #     r"$^{\circ}$",
    #     "K",
    #     0.0001,
    # )

    # reprocess_paramspace(
    #     "dual_omega_starttemp",
    #     os.path.curdir,
    #     "omega",
    #     "starttemp",
    #     "days$^{-1}$",
    #     "K",
    #     0.0001,
    # )
