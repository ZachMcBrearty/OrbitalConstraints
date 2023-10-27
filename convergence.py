from typing import Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from ClimateModel import climate_model_in_lat
from filemanagement import load_config, CONF_PARSER_TYPE


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
    val_unit: Optional[str] = None,
):
    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit

    def t(
        conf: CONF_PARSER_TYPE,
        val_min: float,
        val_max: float,
        val_step: float,
        rtol=0.0001,
    ) -> tuple[NDArray[np.floating], list[float], list[float]]:
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

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.scatter(val_range, tests)
            ax2.scatter(val_range, convtemps)
            ax2.set_xlabel(f"{val_name} {val_unit}")
            ax1.set_xticks(np.linspace(min(val_range), max(val_range), 11))
            ax2.set_xticks(np.linspace(min(val_range), max(val_range), 11))
            ax1.set_ylabel("Time to converge, years")
            ax2.set_ylabel("Global convergent temperature, K")
            plt.show()

        return val_range, tests, convtemps

    return t


test_spacedim_convergence = gen_convergence_test("PDE", "spacedim", True, True, None)
test_timestep_convergence = gen_convergence_test("PDE", "timestep", True, True, "days")
test_temp_convergence = gen_convergence_test("PDE", "start_temp", True, True, "K")

test_omega_convergence = gen_convergence_test(
    "PLANET", "omega", True, True, "day$^{-1}$"
)
test_delta_convergence = gen_convergence_test(
    "PLANET", "obliquity", True, True, r"$^{\circ}"
)

test_a_convergence = gen_convergence_test("ORBIT", "a", True, True, "AU")
test_e_convergence = gen_convergence_test("ORBIT", "e", True, True, None)


def gen_paramspace(
    val_sec_1: str,
    val_name_1: str,
    val_sec_2: str,
    val_name_2: str,
    verbose=False,
    plot=False,
    val_unit_1: Optional[str] = None,
    val_unit_2: Optional[str] = None,
):
    if val_unit_1 is None:
        val_unit_1 = ""
    else:
        val_unit_1 = ", " + val_unit_1
    if val_unit_2 is None:
        val_unit_2 = ""
    else:
        val_unit_2 = ", " + val_unit_2

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
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1: plt.Axes
            ax2: plt.Axes
            converge_time_map = ax1.pcolormesh(
                val_1_range, val_2_range, tests.T, cmap="RdBu_r", shading="nearest"
            )
            converge_temp_map = ax2.pcolormesh(
                val_1_range, val_2_range, convtemps.T, cmap="RdBu_r", shading="nearest"
            )
            fig.colorbar(converge_time_map, ax=ax1, label="Time to converge, years")
            fig.colorbar(converge_temp_map, ax=ax2, label="Convergent Temperature, K")
            ax1.set_ylabel(f"{val_name_2} {val_unit_2}")
            ax1.set_xlabel(f"{val_name_1} {val_unit_1}")
            ax2.set_ylabel(f"{val_name_2} {val_unit_2}")
            ax2.set_xlabel(f"{val_name_1} {val_unit_1}")
            plt.show()

        return val_1_range, val_2_range, tests, convtemps

    return t


dual_a_e_convergence = gen_paramspace(
    "ORBIT", "a", "ORBIT", "e", True, True, "AU", None
)

if __name__ == "__main__":
    conf = load_config()

    conf.set("FILEMANAGEMENT", "save", "False")
    conf.set("FILEMANAGEMENT", "plot", "False")

    conf.set("PDE", "spacedim", "60")  #
    conf.set("PDE", "time", "200")
    conf.set("PDE", "timestep", "1")  #
    conf.set("PDE", "start_temp", "350")  #

    conf.set("PLANET", "omega", "1")  #
    conf.set("PLANET", "land_frac_type", "uniform:0.7")
    conf.set("PLANET", "obliquity", "23.5")  #

    conf.set("ORBIT", "a", "1")  #
    conf.set("ORBIT", "e", "0")  #

    # print(test_a_convergence(conf, 0.5, 2.05, 0.1, rtol=0.0001, plot=True))
    # print(test_e_convergence(conf, 0, 0.91, 0.1, rtol=0.0001, plot=True))
    # print(test_delta_convergence(conf, 0, 181, 10, rtol=0.0001, plot=True))
    # print(test_omega_convergence(conf, 0.3, 3, 0.3, rtol=0.0001, plot=True))
    # print(test_temp_convergence(conf, 100, 501, 50, rtol=0.0001))
    # print(test_spacedim_convergence(conf, 140, 150, 1, rtol=0.0001))
    # print(test_timestep_convergence(conf, 0.25, 3.1, 0.25, rtol=0.0001))
    # print(test_timestep_convergence(conf, 3, 10.1, 0.5, rtol=0.0001))

    print(dual_a_e_convergence(conf, 0.6, 2.05, 0.2, 0, 0.91, 0.3, 0.001))
