from typing import Optional, Literal

import matplotlib.pyplot as plt
from matplotlib import markers
import numpy as np
from scipy.optimize import curve_fit

from Constants import *


def complexplotdata(degs, Temp, dt, Ir_emission, Source, Albedo, Capacity):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    q1 = 0
    q2 = len(degs)
    for n in range(0, 50, 5):
        ax1.plot(
            degs[q1:q2],
            Temp[q1:q2, n],
            label=f"t={dt * n :.3f}",
        )
        ax2.plot(
            degs[q1:q2],
            -Ir_emission[q1:q2, n] + Source[q1:q2, n] * (1 - Albedo[q1:q2, n]),
        )
        ax3.plot(
            degs[q1:q2],
            (Temp[q1:q2, n + 1] - Temp[q1:q2, n])
            * Capacity[q1:q2, n]
            / (YEARTOSECOND * dt)
            - (-Ir_emission[q1:q2, n] + Source[q1:q2, n] * (1 - Albedo[q1:q2, n])),
        )
        ax4.plot(degs[q1:q2], -Ir_emission[q1:q2, n])
        ax5.plot(degs[q1:q2], Source[q1:q2, n] * (1 - Albedo[q1:q2, n]))
        # ax6.plot(degs[q1:q2], 1 / Capacity[q1:q2, n])
        ax6.plot(degs[q1:q2], Source[q1:q2, n])
    ax1.set_ylabel("Temp, K")
    ax2.set_ylabel("-I + S(1-A)")
    ax3.set_ylabel("Diff_elem")
    ax4.set_ylabel("-I")
    ax5.set_ylabel("S(1-A)")
    ax6.set_ylabel("S")
    plt.show()


def yearavgplot(degs, temp, dt, start_yr=0, end_yr=None, year_skip=1):
    if end_yr is None:
        end_yr = len(temp[0, :]) // 365
    fig, (ax, ax2) = plt.subplots(2, 1, height_ratios=[2, 1])
    for n in range(start_yr, end_yr, year_skip):
        yr_avg = np.average(temp[:, n * 365 : (n + 1) * 365], axis=1)
        a = ax.plot(degs, yr_avg, label=f"EBCM")
    # a[0].set_marker("x")  # type:ignore
    ax.plot(
        degs,
        302.3 - 45.3 * np.sin(np.deg2rad(degs)) ** 2,
        # marker=".",
        ls="--",
        label="NC97",
    )
    ax2.scatter(degs, (yr_avg - (302.3 - 45.3 * np.sin(np.deg2rad(degs)) ** 2)))  # type: ignore
    ax.axhline(273, ls="--", label=r"0$^\circ$C")
    ax.set_ylabel("Average Temperature, K")
    ax2.set_ylabel("EBCM - NC97, K")
    ax2.set_xlabel(r"$\lambda$")
    ax.set_xticks(range(-90, 91, 15))
    ax2.set_xticks(range(-90, 91, 15))
    ax.legend()
    plt.show()


def plotdata(degs, temp, dt, start=0, end=None, numplot=10):
    if end is None:
        end = len(temp[0, :])
    for n in range(start, end, (end - start) // numplot):
        a = plt.plot(degs, temp[:, n], label=f"t={dt * n :.3f} yrs")
    plt.axhline(273, ls="--", label=r"0$^\circ$C")
    plt.ylabel("Temperature, K")
    plt.xlabel(r"$\lambda$, degrees")
    plt.xticks(list(range(-90, 91, 15)))
    plt.legend()
    plt.show()


def colourplot(
    degs,
    temps,
    times,
    yr_start=None,
    yr_end=None,
    # year_avg=1,
    lat_start=None,
    lat_end=None,
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

    # time average
    # tq = np.average(temp.reshape((temp.shape[0], -1, 365 * year_avg)), axis=2)

    pcm = ax.pcolormesh(
        ts, degs[lat_start:lat_end], temp, cmap=cmap, shading="nearest"
    )  # nearest

    ax.set_xlabel("time, yr")
    ax.set_ylabel("latitude, degrees")
    ax.set_yticks(np.linspace(degs[lat_start], degs[lat_end - 1], 13, endpoint=True))
    fig.colorbar(pcm, ax=ax, label="Temperature, K")
    plt.tight_layout()
    plt.show()


def threecolourplot(
    plt1,
    plt2,
    plt3,
    yr_start=None,
    yr_end=None,
    year_avg=1,
):
    if yr_start is None:
        yr_start = 0
    else:
        yr_start *= 365
    if yr_end is None:
        yr_end = plt1[1].shape[1]
    else:
        yr_end = (yr_end + 1) * 365

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes
    cmap = "RdBu_r"
    ts = plt1[0][yr_start : yr_end : year_avg * 365]

    temp1 = plt1[1][:, yr_start + 1 : yr_end + 1]
    temp2 = plt2[1][:, yr_start + 1 : yr_end + 1]
    temp3 = plt3[1][:, yr_start + 1 : yr_end + 1]

    # time average
    tq1 = np.average(temp1.reshape((temp1.shape[0], -1, 365 * year_avg)), axis=2)
    tq2 = np.average(temp2.reshape((temp2.shape[0], -1, 365 * year_avg)), axis=2)
    tq3 = np.average(temp3.reshape((temp3.shape[0], -1, 365 * year_avg)), axis=2)

    _min, _max = np.amin((tq1, tq2, tq3)), np.amax((tq1, tq2, tq3))
    pcm1 = ax1.pcolormesh(
        ts, plt1[2], tq1, cmap=cmap, shading="nearest", vmin=_min, vmax=_max
    )  # nearest
    pcm2 = ax2.pcolormesh(
        ts, plt2[2], tq2, cmap=cmap, shading="nearest", vmin=_min, vmax=_max
    )  # nearest
    pcm3 = ax3.pcolormesh(
        ts, plt3[2], tq3, cmap=cmap, shading="nearest", vmin=_min, vmax=_max
    )  # nearest

    ax3.set_xlabel("time, yr")
    # ax1.set_ylabel("latitude, degrees")
    ax2.set_ylabel("latitude, degrees")
    # ax3.set_ylabel("latitude, degrees")
    ax1.set_yticks(np.linspace(-90, 90, 7, endpoint=True))
    ax2.set_yticks(np.linspace(-90, 90, 7, endpoint=True))
    ax3.set_yticks(np.linspace(-90, 90, 7, endpoint=True))

    fig.colorbar(pcm1, ax=(ax1, ax2, ax3), label="Temperature, T, K")

    # plt.tight_layout()
    plt.show()


def convergence_plot_single(
    tests: np.ndarray,
    convtemps: np.ndarray,
    val_name: str,
    val_range,
    val_unit: Optional[str] = None,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit

    fig, ax2 = plt.subplots(1, 1)
    ax2: plt.Axes

    if min(convtemps) < 273:
        ax2.axhline(273, 0, 1, ls="-.", label="273 K")
    if max(convtemps) > 373:
        ax2.axhline(373, 0, 1, ls="-.", label="373 K")
    # tests[convtemps < 0] = np.nan
    # convtemps[convtemps < 0] = np.nan

    # ax2.axvline(22, 0, 1, label="Earth min, max")
    # ax2.axvline(25, 0, 1)

    ax2.scatter(val_range, convtemps)

    ax2.set_xlabel(f"{val_name} {val_unit}")

    ax2.set_xscale(x_axis_scale)
    ax2.set_yscale(y_axis_scale)

    ax2.set_xticks(np.linspace(val_range[0], val_range[-1], 11))

    ax2.set_ylabel(global_conv_temp_name + ", " + temp_unit)
    ax2.legend()
    plt.show()


def single_variable_single_fit_plot(
    tests: NDArray,
    convtemps: NDArray,
    val_name: str,
    val_range: NDArray,
    model_function,
    model_range: tuple[int, int] = (0, -1),
    model_function_label: str = "Fitting model",
    val_unit: Optional[str] = None,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
    residuals: bool = False,
):
    """model_function: f(xdata, *params)"""
    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit
    if residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=(2, 1), sharex="all")
    else:
        fig, ax1 = plt.subplots(1, 1)
        ax2 = ax1
    ax1: plt.Axes
    ax2: plt.Axes

    if min(convtemps) < 273:
        ax1.axhline(273, 0, 1, ls="-.", label="273 K")
    if max(convtemps) > 373:
        ax1.axhline(373, 0, 1, ls="-.", label="373 K")
    # plot the model data on the main graph
    ax1.scatter(val_range, convtemps, c="b", marker=markers.MarkerStyle("x"))
    # find model parameters
    fit_parameters, pcov = curve_fit(
        model_function,
        val_range[model_range[0] : model_range[1]],
        convtemps[model_range[0] : model_range[1]],
    )
    print(
        f"Model: {model_function_label}: {fit_parameters} +/- {np.sqrt(np.diag(pcov))}"
    )
    fit_xs = np.linspace(val_range[model_range[0]], val_range[model_range[1]], 101)
    fit_ys = model_function(fit_xs, *fit_parameters)
    # plot the model on the main graph
    ax1.plot(fit_xs, fit_ys, c="r", label=model_function_label)

    # plot residuals on secondary graph
    if residuals:
        ax2.scatter(
            val_range[model_range[0] : model_range[1]],
            convtemps[model_range[0] : model_range[1]]
            - model_function(
                val_range[model_range[0] : model_range[1]], *fit_parameters
            ),
            c="b",
            marker=markers.MarkerStyle("x"),
        )

    # ax1.axvline(0.0167, 0, 1, label="Earth current")
    # ax1.axvline(0.0679, 0, 1, label="Earth max")
    # share-x is used so only the bottom graph needs to have the x-label
    if residuals:
        ax2.set_xlabel(f"{val_name}{val_unit}")
        ax2.set_ylabel("Residual, K")
        ax2.set_xscale(x_axis_scale)
        ax2.set_xticks(np.linspace(val_range[0], val_range[-1], 11))
    else:
        ax1.set_xlabel(f"{val_name}{val_unit}")
    ax1.set_ylabel(global_conv_temp_name + ", " + temp_unit)
    # use user selected x-axis scale for both
    ax1.set_xscale(x_axis_scale)
    # use user selected y-axis only for the main plot as residual is always linear
    ax1.set_yscale(y_axis_scale)
    ax1.set_xticks(np.linspace(val_range[0], val_range[-1], 11))

    ax1.legend()

    plt.tight_layout()
    plt.show()


def single_variable_N_fits_plot(
    tests: NDArray,
    convtemps: NDArray,
    val_name: str,
    val_range: NDArray,
    model_functions: list,
    model_ranges: list[tuple[int, int]],
    model_function_labels: list[str],
    val_unit: Optional[str] = None,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
    residuals: bool = False,
):
    """model_function: f(xdata, *params)"""
    assert len(model_functions) == len(model_function_labels) == len(model_ranges)
    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit
    if residuals:
        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=(2, 1), sharex="all")
    else:
        fig, ax1 = plt.subplots(1, 1)
        ax2 = ax1
    ax1: plt.Axes
    ax2: plt.Axes
    if min(convtemps) < 273:
        ax1.axhline(273, 0, 1, ls="-.", label="273 K")
    if max(convtemps) > 373:
        ax1.axhline(373, 0, 1, ls="-.", label="373 K")
    # plot the model data on the main graph
    ax1.scatter(val_range, convtemps, c="b", marker=markers.MarkerStyle("x"))
    # find model parameters
    for i in range(len(model_functions)):
        model_function = model_functions[i]
        model_range = model_ranges[i]
        model_function_label = model_function_labels[i]

        fit_parameters, pcov = curve_fit(
            model_function,
            val_range[model_range[0] : model_range[1]],
            convtemps[model_range[0] : model_range[1]],
        )
        print(
            f"Model {i}: {model_function_label}: {fit_parameters} +/- {np.sqrt(np.diag(pcov))}"
        )
        if model_range[1] == -1:
            fit_xs = np.linspace(
                val_range[model_range[0]], val_range[model_range[1]], 101
            )
        else:
            fit_xs = np.linspace(
                val_range[model_range[0]], val_range[model_range[1] - 1], 101
            )
        fit_ys = model_function(fit_xs, *fit_parameters)
        # plot the model on the main graph
        ax1.plot(fit_xs, fit_ys, c="r", label=model_function_label)

        # plot residuals on secondary graph
        if residuals:
            ax2.scatter(
                val_range[model_range[0] : model_range[1]],
                convtemps[model_range[0] : model_range[1]]
                - model_function(
                    val_range[model_range[0] : model_range[1]], *fit_parameters
                ),
                c="b",
                marker=markers.MarkerStyle("x"),
            )
    ax1.axvline(DISTANCE["venus"] / AU, 0, 1, label="Venus")
    ax1.axvline(DISTANCE["mars"] / AU, 0, 1, label="Mars")
    # share-x is used so only the bottom graph needs to have the x-label
    if residuals:
        ax2.set_xlabel(f"{val_name}{val_unit}")
        ax2.set_ylabel("Residual, K")
        ax2.set_xscale(x_axis_scale)
        ax2.set_xticks(np.linspace(val_range[0], val_range[-1], 11))
    else:
        ax1.set_xlabel(f"{val_name}{val_unit}")
    ax1.set_ylabel(global_conv_temp_name + ", " + temp_unit)
    # use user selected x-axis scale for both
    ax1.set_xscale(x_axis_scale)
    # use user selected y-axis only for the main plot as residual is always linear
    ax1.set_yscale(y_axis_scale)
    ax1.set_xticks(np.linspace(val_range[0], val_range[-1], 11))

    ax1.legend()

    plt.tight_layout()
    plt.show()


def convergence_plot_single_compare(
    tests_1,
    convtemps_1,
    val_name_1: str,
    val_range_1,
    tests_2,
    convtemps_2,
    val_name_2: str,
    val_range_2,
    val_unit: Optional[str] = None,
):
    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(val_range_1, tests_1, label=f"{val_name_1}")
    ax2.scatter(val_range_1, convtemps_1, label=f"{val_name_1}")

    ax1.scatter(val_range_2, tests_2, label=f"{val_name_2}")
    ax2.scatter(val_range_2, convtemps_2, label=f"{val_name_2}")

    if val_name_1 == val_name_2:
        ax2.set_xlabel(f"{val_name_1} {val_unit}")
    else:
        ax2.set_xlabel(f"{val_name_1} and {val_name_2} {val_unit}")

    ax1.set_ylabel("Time to converge, years")
    ax2.set_ylabel(global_conv_temp_name + ", " + temp_unit)
    ax1.legend()
    ax2.legend()
    plt.show()


def convergence_plot_dual(
    tests,
    convtemps,
    val_name_1,
    val_1_range,
    val_name_2,
    val_2_range,
    val_unit_1=None,
    val_unit_2=None,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    if val_unit_1 is None:
        val_unit_1 = ""
    else:
        val_unit_1 = ", " + val_unit_1
    if val_unit_2 is None:
        val_unit_2 = ""
    else:
        val_unit_2 = ", " + val_unit_2

    fig, ax2 = plt.subplots(1, 1)
    ax2: plt.Axes
    converge_temp_map = ax2.pcolormesh(
        val_1_range, val_2_range, convtemps.T, cmap="RdBu_r", shading="nearest"
    )

    # roche = 0.00042492578167103155  # au
    # for i, semi in enumerate(val_1_range):
    #     for j, ecc in enumerate(val_2_range):
    #         if semi * (1 - ecc) < roche:
    #             print("rochelimit!", i, j, ecc, semi)

    fig.colorbar(
        converge_temp_map, ax=ax2, label=global_conv_temp_name + ", " + temp_unit
    )
    ax2.set_xlabel(f"{val_name_1} {val_unit_1}")
    ax2.set_ylabel(f"{val_name_2} {val_unit_2}")

    ax2.set_xticks(val_1_range)
    ax2.set_yticks(val_2_range)

    ax2.set_xscale(x_axis_scale)
    ax2.set_yscale(y_axis_scale)

    plt.show()


def convergence_plot_dual_compare(
    tests_1,
    convtemps_1,
    tests_2,
    convtemps_2,
    val_name_1,
    val_1_range_1,
    val_1_range_2,
    val_name_2,
    val_2_range_1,
    val_2_range_2,
    val_unit_1=None,
    val_unit_2=None,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    if val_unit_1 is None:
        val_unit_1 = ""
    else:
        val_unit_1 = ", " + val_unit_1
    if val_unit_2 is None:
        val_unit_2 = ""
    else:
        val_unit_2 = ", " + val_unit_2

    fig, (ax2, ax3) = plt.subplots(2, 1)
    ax2: plt.Axes
    ax3: plt.Axes
    converge_temp_map_1 = ax2.pcolormesh(
        val_1_range_1, val_2_range_1, convtemps_1.T, cmap="RdBu_r", shading="nearest"
    )
    converge_temp_map_2 = ax3.pcolormesh(
        val_1_range_2, val_2_range_2, convtemps_2.T, cmap="RdBu_r", shading="nearest"
    )

    fig.colorbar(
        converge_temp_map_1, ax=ax2, label=global_conv_temp_name + ", " + temp_unit
    )
    fig.colorbar(
        converge_temp_map_2, ax=ax3, label=global_conv_temp_name + ", " + temp_unit
    )

    ax2.set_ylabel(f"{val_name_2} {val_unit_2}")
    ax2.set_xlabel(f"{val_name_1} {val_unit_1}")
    ax3.set_ylabel(f"{val_name_2} {val_unit_2}")
    ax3.set_xlabel(f"{val_name_1} {val_unit_1}")

    ax2.set_xticks(val_1_range_1)
    ax2.set_yticks(val_2_range_1)
    ax3.set_xticks(val_1_range_2)
    ax3.set_yticks(val_2_range_2)

    ax2.set_xscale(x_axis_scale)
    ax2.set_yscale(y_axis_scale)
    ax3.set_xscale(x_axis_scale)
    ax3.set_yscale(y_axis_scale)

    plt.show()


def double_variable_single_fit_plot():
    pass


def double_variable_N_fits_plot():
    pass


def convergence_plot_dual_with_fits(
    tests: np.ndarray,
    convtemps: np.ndarray,
    val_name_1,
    val_1_range,
    val_name_2,
    val_2_range,
    val_unit_1=None,
    val_unit_2=None,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    if val_unit_1 is None:
        val_unit_1 = ""
    else:
        val_unit_1 = ", " + val_unit_1
    if val_unit_2 is None:
        val_unit_2 = ""
    else:
        val_unit_2 = ", " + val_unit_2

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    fig, ax2 = plt.subplots(1, 1)
    # ax1: plt.Axes
    ax2: plt.Axes
    # converge_time_map = ax1.pcolormesh(
    #     val_1_range, val_2_range, tests.T, cmap="RdBu_r", shading="nearest"
    # )
    converge_temp_map = ax2.pcolormesh(
        val_1_range, val_2_range, convtemps.T, cmap="RdBu_r", shading="nearest"
    )

    # fig.colorbar(converge_time_map, ax=ax1, label="Time to converge, years")
    fig.colorbar(
        converge_temp_map, ax=ax2, label=global_conv_temp_name + ", " + temp_unit
    )

    fit_xs = []
    fit_ys = []
    q = 25
    for x, tx in enumerate(convtemps):
        for y, t in enumerate(tx):
            if 273 - q < t < 273 + q:
                print("First", x, val_1_range[x], y, val_2_range[y], t)
                fit_xs.append(val_1_range[x])
                fit_ys.append(val_2_range[y])
    xs = np.linspace(fit_xs[0], fit_xs[-1], 100)
    k = 1 / ((1 - fit_ys[0] ** 2) * fit_xs[0] ** 4)
    fit = lambda x: (1 - 1 / (k * x**4)) ** (1 / 2)

    ys = fit(xs)
    ax2.scatter(fit_xs, fit_ys, c="r", label=f"{273-q} K $ < T < $ {273+q} K")
    ax2.plot(xs, ys, c="r", label=r"$a \propto (1-e^2)^{-1/4}, T \approx 273$K")

    fit_xs = []
    fit_ys = []
    q = 12
    for x, tx in enumerate(convtemps):
        for y, t in enumerate(tx):
            if 373 - q < t < 373 + q:
                print("Second", x, val_1_range[x], y, val_2_range[y], t)
                fit_xs.append(val_1_range[x])
                fit_ys.append(val_2_range[y])
    # fitter = lambda x, a: a * x**(15/4)
    xs = np.linspace(fit_xs[0], fit_xs[-1], 100)
    k = 1 / ((1 - fit_ys[0] ** 2) * fit_xs[0] ** 4)
    fit = lambda x: (1 - 1 / (k * x**4)) ** (1 / 2)
    ys = fit(xs)
    ax2.scatter(fit_xs, fit_ys, c="b", label=f"{373-q} K $ < T < $ {373+q} K")
    ax2.plot(xs, ys, c="b", label=r"$a \propto (1-e^2)^{-1/4}, T \approx 373$K")
    # ax2.plot(xs, ys_prime)
    # ax2.scatter(fit_val_1_max, fit_val_2_max, label="max")

    # ax1.set_ylabel(f"{val_name_2} {val_unit_2}")
    # ax1.set_xlabel(f"{val_name_1} {val_unit_1}")
    ax2.set_ylabel(f"{val_name_2} {val_unit_2}")
    ax2.set_xlabel(f"{val_name_1} {val_unit_1}")

    # ax1.set_xticks(val_1_range)
    # ax1.set_yticks(val_2_range)
    ax2.set_xticks(val_1_range)
    ax2.set_yticks(val_2_range)
    # ax1.set_xscale(x_axis_scale)
    # ax1.set_yscale(y_axis_scale)
    ax2.set_xscale(x_axis_scale)
    ax2.set_yscale(y_axis_scale)

    ax2.legend(loc="lower right")
    plt.show()

def plot_three_shared_x_axis(
    tests,
    convtemps,
    val_ranges,
    val_name,
    val_unit,
    series_labels,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes: list[plt.Axes]
    for i in range(len(convtemps)):
        axes[i].scatter(val_ranges[i], convtemps[i], label=series_labels[i])

    # if np.min(convtemps) < 273:
    #     ax1.axhline(273, 0, 1, ls="-.", label="273 K")
    # if np.max(convtemps) > 373:
    #     ax1.axhline(373, 0, 1, ls="-.", label="373 K")
    axes[2].set_xlabel(f"{val_name}{val_unit}")
    
    axes[1].set_ylabel(global_conv_temp_name+", "+temp_unit)

    for i in range(3):
        axes[i].set_xscale(x_axis_scale)
        axes[i].set_yscale(y_axis_scale)
        axes[i].legend()

    axes[2].set_xticks(np.linspace(min(val_ranges[2]), max(val_ranges[2]), 11))
    plt.tight_layout()
    plt.show()

def plot_three_on_one_graph(
    tests,
    convtemps,
    val_ranges,
    val_name,
    val_unit,
    series_labels,
    x_axis_scale: Literal["linear", "log"] = "linear",
    y_axis_scale: Literal["linear", "log"] = "linear",
):
    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit

    fig, ax = plt.subplots()
    ax: plt.Axes
    for i in range(len(convtemps)):
        ax.scatter(val_ranges[i], convtemps[i], label=series_labels[i])

    if np.min(convtemps) < 273:
        ax.axhline(273, 0, 1, ls="-.", label="273 K")
    if np.max(convtemps) > 373:
        ax.axhline(373, 0, 1, ls="-.", label="373 K")
    ax.set_xlabel(f"{val_name}{val_unit}")
    
    ax.set_ylabel(global_conv_temp_name+", "+temp_unit)

    ax.set_xscale(x_axis_scale)
    ax.set_yscale(y_axis_scale)

    plt.legend()
    plt.tight_layout()
    plt.show()


def orbital_animation():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from orbital_model import orbital_model, line, r
    from filemanagement import load_config

    config = load_config("config.ini", "OrbitalConstraints")
    r_star = config.getfloat("ORBIT", "starradius")
    r_gas = config.getfloat("ORBIT", "gasgiantradius")
    n = 24
    poses = orbital_model(config, dt_steps=n)
    # for x in range(15000):
    #     next(poses)
    fig, ax = plt.subplots()
    ax: plt.Axes
    fig.set_figheight(5)
    fig.set_figwidth(5)
    star, gas, moon = [0, 0], [0, 0], [0, 0]
    (ln1,) = ax.plot(0, 0, "ro", ms=5)
    (ln2,) = ax.plot(0, 0, "bo", ms=5)
    (ln3,) = ax.plot(0, 0, "go", ms=5)
    text = ax.text(0.75, 0.9, f"Eclipsed: {1}", transform=ax.transAxes)

    (toptop,) = ax.plot([0, 0], [1, 1], "b-")
    (bottombottom,) = ax.plot([0, 0], [-1, -1], "b-")
    (topbottom,) = ax.plot([0, 0], [1, -1], "g-")
    (bottomtop,) = ax.plot([0, 0], [-1, 1], "g-")

    def init():
        return (ln1, ln2, ln3, text, toptop, bottombottom, topbottom, bottomtop)

    def animate(i):
        eclip = 0
        star, gas, moon, eclipsed = next(poses)
        eclip += eclipsed
        for _ in range(n - 1):
            star, gas, moon, eclipsed = next(poses)
            eclip += eclipsed
        center = gas
        eclip /= n
        ln1.set_data(star - center)
        ln2.set_data(gas - center)
        ln3.set_data(moon - center)
        text.set_text(f"Eclipsed: {eclip}")
        star_to_gas_dir = (gas - star) / r(gas - star)
        perp_up = np.array([star_to_gas_dir[1], -star_to_gas_dir[0]])
        perp_down = np.array([-star_to_gas_dir[1], star_to_gas_dir[0]])
        q = star + r_star * perp_up
        p = gas + r_gas * perp_up
        toptop.set_data(
            [q[0] - center[0], p[0] - center[0], 2 * p[0] - center[0]],  # moon[0]],
            [
                q[1] - center[1],
                p[1] - center[1],
                line(2 * p[0], q, p) - center[1],
            ],  # line(moon[0], q, p)],
        )
        q = star + r_star * perp_down
        p = gas + r_gas * perp_down
        bottombottom.set_data(
            [q[0] - center[0], p[0] - center[0], 2 * p[0] - center[0]],  # moon[0]],
            [
                q[1] - center[1],
                p[1] - center[1],
                line(2 * p[0], q, p) - center[1],
            ],  # line(moon[0], q, p)],
        )
        q = star + r_star * perp_up
        p = gas + r_gas * perp_down
        topbottom.set_data(
            [q[0] - center[0], p[0] - center[0], 2 * p[0] - center[0]],  # moon[0]],
            [
                q[1] - center[1],
                p[1] - center[1],
                line(2 * p[0], q, p) - center[1],
            ],  # line(moon[0], q, p)],
        )
        q = star + r_star * perp_down
        p = gas + r_gas * perp_up
        bottomtop.set_data(
            [q[0] - center[0], p[0] - center[0], 2 * p[0] - center[0]],  # moon[0]],
            [
                q[1] - center[1],
                p[1] - center[1],
                line(2 * p[0], q, p) - center[1],
            ],  # line(moon[0], q, p)],
        )
        return (ln1, ln2, ln3, text, toptop, bottombottom, topbottom, bottomtop)

    bound = 1.2 * 1.5 * 10**11
    ax.set_xbound(-bound, bound)
    ax.set_ybound(-bound, bound)

    ani = FuncAnimation(fig, animate, frames=range(0, 1000), init_func=init, blit=True)
    plt.show()


if __name__ == "__main__":
    from filemanagement import load_config, read_file, read_single_folder
    from Habitabilites import habitabilitycolourplot, HumanCompatible, Habitable
    from convergence import convergence_test

    # times, temps, degs = read_file("Earth.npz")
    # habitabilitycolourplot(degs, temps, times, 190, 192, None, None, Habitable, "(LWR)")
    # times, temps, degs = read_file("Earth.npz")
    # habitabilitycolourplot(
    #     degs, temps, times, 190, 192, None, None, HumanCompatible, "(HC)"
    # )
    # conf = load_config()
    # q0 = read_file("single_omega/single_omega_0.9375.npz")
    # q1 = read_file("single_omega/single_omega_1.0.npz")
    # q2 = read_file("single_omega/single_omega_1.0625.npz")
    # threecolourplot(q0, q1, q2, None, None, 1)

    # q0 = read_file("single_landfractype/single_landfractype_uniform_0.160.npz")
    # # 680 700 720
    # # 0.160 , 180 , 200
    # q1 = read_file("single_landfractype/single_landfractype_uniform_0.180.npz")
    # q2 = read_file("single_landfractype/single_landfractype_uniform_0.200.npz")
    # threecolourplot(q0, q1, q2, 100, None, 1)

    # q0 = read_file("single_omega/single_omega_2.25.npz")
    # q1 = read_file("single_omega/single_omega_2.3125.npz")
    # q2 = read_file("single_omega/single_omega_2.375.npz")
    # threecolourplot(q0, q1, q2, None, None, 1)

    # times, temps, degs = read_file(
    #     "dual_gassemimajoraxis_gaseccentricity/dual_gassemimajoraxis_1.0_gaseccentricity_0.0.npz"
    # )
    # dt = times[1] - times[0]
    # # colourplot(degs, temps, times, 90, None, 1)
    # # plotdata(degs, temps, dt, int(365 * 90.5), int(365 * 91.5), 12)
    # # print(convergence_test(temps, rtol=0.0001))
    # yearavgplot(degs, temps, dt, 90, 100, 10)

    # colourplot(degs, temps, times, None, None, 1, None, None)

    # one = read_files("testing_1.5.npz")
    # two = read_files("testing_3.npz")
    # three = read_files("testing_6.npz")

    # threecolourplot(one, two, three, 90, 120, 1)
