from typing import Optional
import os

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from ClimateModel import climate_model_in_lat
from filemanagement import (
    load_config,
    write_to_file,
    read_dual_folder,
    CONF_PARSER_TYPE,
)
from plotting import convergence_plot_single, convergence_plot_dual


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
    a = int(temps.size % (year_len * 60 * year_avg) / 60)
    tq = temps[:, a:].reshape(temps.shape[0], -1, year_len * year_avg)
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
        val_min: float,
        val_max: float,
        val_step: float,
        rtol=0.0001,
    ) -> tuple[NDArray[np.floating], list[float], list[float]]:
        if not conf.has_option(val_section, val_name):
            raise ValueError(
                f"Supplied config file has no section {val_section} or option {val_name}"
            )
        tests = []
        convtemps = []
        val_range = np.arange(val_min, val_max, val_step)
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
                filename = f"./single_{val_name}/single_{val_name}_{round(val,3)}.npz"
                write_to_file(
                    times,
                    temps,
                    degs,
                    filename,
                )

        if plot:
            convergence_plot_single(
                tests, convtemps, val_name, val_min, val_max, val_step, val_unit
            )

        return val_range, tests, convtemps

    return t


test_spacedim_convergence = gen_convergence_test(
    "PDE", "spacedim", True, True, True, None
)
test_timestep_convergence = gen_convergence_test(
    "PDE", "timestep", True, True, True, "days"
)
test_temp_convergence = gen_convergence_test("PDE", "starttemp", True, True, True, "K")

test_omega_convergence = gen_convergence_test(
    "PLANET", "omega", True, True, True, "day$^{-1}$"
)
test_delta_convergence = gen_convergence_test(
    "PLANET", "obliquity", True, True, True, r"$^{\circ}"
)

test_a_convergence = gen_convergence_test("ORBIT", "a", True, True, True, "AU")
test_e_convergence = gen_convergence_test("ORBIT", "e", True, True, True, None)


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
        val_min_1: float,
        val_max_1: float,
        val_step_1: float,
        val_min_2: float,
        val_max_2: float,
        val_step_2: float,
        rtol=0.0001,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        if not conf.has_option(val_sec_1, val_name_1):
            raise ValueError(
                f"Supplied config file has no section {val_sec_1} or option {val_name_1}"
            )
        if not conf.has_option(val_sec_2, val_name_2):
            raise ValueError(
                f"Supplied config file has no section {val_sec_2} or option {val_name_2}"
            )
        val_1_range = np.arange(val_min_1, val_max_1, val_step_1)
        val_2_range = np.arange(val_min_2, val_max_2, val_step_2)
        tests = np.zeros((len(val_1_range), len(val_2_range)))
        convtemps = np.zeros_like(tests)
        for i, val_1 in enumerate(val_1_range):
            for j, val_2 in enumerate(val_2_range):
                if verbose:
                    print(f"Running {val_name_1}={val_1}, {val_name_2}={val_2}")
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
                    filename = f"./dual_{val_name_1}_{val_name_2}/dual_{val_name_1}_{round(val_1,3)}_{val_name_2}_{round(val_2, 3)}.npz"
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
    val_unit_1="AU",
    val_unit_2=None,
    save=True,
)
dual_a_delta_convergence = gen_paramspace(
    "ORBIT",
    "a",
    "PLANET",
    "obliquity",
    verbose=True,
    plot=True,
    val_unit_1="AU",
    val_unit_2=r"$^{\circ}$",
    save=True,
)

dual_a_omega_convergence = gen_paramspace(
    "ORBIT",
    "a",
    "PLANET",
    "omega",
    True,
    True,
    "AU",
    "days$^-1$",
    True,
)

dual_a_temp_convergence = gen_paramspace(
    "ORBIT",
    "a",
    "PDE",
    "starttemp",
    True,
    True,
    "AU",
    "K",
    True,
)

dual_e_delta_convergence = gen_paramspace(
    "ORBIT", "e", "PLANET", "obliquity", True, True, None, r"$^{\circ}$", True
)

dual_e_omega_convergence = gen_paramspace(
    "ORBIT", "e", "PLANET", "omega", True, True, None, "days$^{-1}$", True
)

dual_e_starttemp_convergence = gen_paramspace(
    "ORBIT", "e", "PDE", "starttemp", True, True, None, "K", True
)

dual_delta_omega_convergence = gen_paramspace(
    "PLANET",
    "obliquity",
    "PLANET",
    "omega",
    True,
    True,
    r"$^{\circ}$",
    "days$^{-1}$",
    True,
)

dual_delta_starttemp_convergence = gen_paramspace(
    "PLANET", "obliquity", "PDE", "starttemp", True, True, r"$^{\circ}$", "K", True
)

dual_omega_starttemp_convergence = gen_paramspace(
    "PLANET", "omega", "PDE", "starttemp", True, True, "days$^{-1}$", "K", True
)


def reprocess_paramspace(
    foldername: str,
    folderpath: str,
    val_1_name: str,
    val_2_name: str,
    val_1_unit: Optional[str] = None,
    val_2_unit: Optional[str] = None,
    rtol: float = 0.0001,
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
        if val_1 not in val_1_range:
            val_1_range.append(val_1)
        if val_2 not in val_2_range:
            val_2_range.append(val_2)
        tests.append(t)
        convtemps.append(temp)
    xl = len(val_1_range)
    yl = len(val_2_range)
    tests = np.array(tests).reshape(xl, yl)
    convtemps = np.array(convtemps).reshape(xl, yl)
    convergence_plot_dual(
        tests,
        convtemps,
        val_1_name,
        val_1_range,
        val_2_name,
        val_2_range,
        rtol,
        val_1_unit,
        val_2_unit,
    )


if __name__ == "__main__":
    conf = load_config()

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

    # print(test_a_convergence(conf, 0.5, 2.05, 0.1, rtol=0.0001, plot=True))
    # print(test_e_convergence(conf, 0, 0.91, 0.1, rtol=0.0001, plot=True))
    print(test_delta_convergence(conf, 0, 181, 10, rtol=0.0001))
    # print(test_omega_convergence(conf, 0.3, 3, 0.3, rtol=0.0001, plot=True))
    # print(test_temp_convergence(conf, 100, 501, 50, rtol=0.0001))
    # print(test_spacedim_convergence(conf, 140, 150, 1, rtol=0.0001))
    # print(test_timestep_convergence(conf, 0.25, 3.1, 0.25, rtol=0.0001))
    # print(test_timestep_convergence(conf, 3, 10.1, 0.5, rtol=0.0001))

    # print(dual_a_e_convergence(conf, 0.5, 2.05, 0.1, 0, 0.91, 0.1, 0.001))
    # print(dual_a_delta_convergence(conf, 0.5, 2.05, 0.1, 0, 91, 10, 0.001))
    # print(dual_a_omega_convergence(conf, 0.5, 2.05, 0.1, 0.25, 3.1, 0.25, 0.001))
    # print(dual_a_temp_convergence(conf, 0.5, 2.05, 0.1, 150, 500, 50, 0.001))

    # print(dual_e_delta_convergence(conf, 0, 0.91, 0.1, 0, 91, 10, 0.001))
    # print(dual_e_omega_convergence(conf, 0, 0.91, 0.1, 0.25, 3.1, 0.25, 0.001))
    # print(dual_e_starttemp_convergence(conf, 0, 0.91, 0.1, 150, 500, 50, 0.001))

    # print(dual_delta_omega_convergence(conf, 70, 91, 10, 0.25, 3.1, 0.25, 0.001))
    # print(dual_delta_starttemp_convergence(conf, 0, 91, 10, 150, 500, 50, 0.001))

    # print(dual_omega_starttemp_convergence(conf, 0.25, 3.1, 0.25, 150, 500, 50, 0.001))

    # reprocess_paramspace("dual_a_e", os.path.curdir, "a", "e", "au", None, 0.0001)
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
