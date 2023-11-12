from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from Constants import yeartosecond


def complexplotdata(degs, Temp, dt, Ir_emission, Source, Albedo, Capacity):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    q1 = 0
    q2 = len(degs)
    for n in range(0, 20, 2):
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
            / (yeartosecond * dt)
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
    ax6.set_ylabel("1/Capacity")
    plt.show()


def yearavgplot(degs, temp, dt, start_yr=0, end_yr=None, year_skip=1):
    if end_yr is None:
        end_yr = len(temp[0, :]) // 365
    fig, (ax, ax2) = plt.subplots(2, 1)
    for n in range(start_yr, end_yr, year_skip):
        yr_avg = np.average(temp[:, n * 365 : (n + 1) * 365], axis=1)
        a = ax.plot(degs, yr_avg, label=f"t={n} yrs")
    a[0].set_marker("x")  # type:ignore
    ax.plot(
        degs,
        302.3 - 45.3 * np.sin(np.deg2rad(degs)) ** 2,
        marker=".",
        ls="--",
        label="fit",
    )
    ax2.plot(degs, np.abs(yr_avg - (302.3 - 45.3 * np.sin(np.deg2rad(degs)) ** 2)))  # type: ignore
    ax.axhline(273, ls="--", label=r"0$^\circ$C")
    ax.set_ylabel("Average Temperature, K")
    ax2.set_ylabel("Variation from fit, K")
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
    plt.xticks(range(-90, 91, 15))
    plt.legend()
    plt.show()


def colourplot(
    degs,
    temps,
    times,
    yr_start=None,
    yr_end=None,
    year_avg=1,
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
    ts = times[yr_start : yr_end : year_avg * 365]
    temp = temps[lat_start:lat_end, yr_start + 1 : yr_end + 1]

    # time average
    tq = np.average(temp.reshape((temp.shape[0], -1, 365 * year_avg)), axis=2)

    pcm = ax.pcolormesh(
        ts, degs[lat_start:lat_end], tq, cmap=cmap, shading="nearest"
    )  # nearest

    ax.set_xlabel("time, yr")
    ax.set_ylabel("latitude, degrees")
    ax.set_yticks(np.linspace(degs[lat_start], degs[lat_end - 1], 12, endpoint=True))
    fig.colorbar(pcm, ax=ax)
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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    cmap = "RdBu_r"
    ts = plt1[0][yr_start : yr_end : year_avg * 365]

    temp1 = plt1[1][:, yr_start + 1 : yr_end + 1]
    temp2 = plt2[1][:, yr_start + 1 : yr_end + 1]
    temp3 = plt3[1][:, yr_start + 1 : yr_end + 1]

    # time average
    tq1 = np.average(temp1.reshape((temp1.shape[0], -1, 365 * year_avg)), axis=2)
    tq2 = np.average(temp2.reshape((temp2.shape[0], -1, 365 * year_avg)), axis=2)
    tq3 = np.average(temp3.reshape((temp3.shape[0], -1, 365 * year_avg)), axis=2)

    pcm1 = ax1.pcolormesh(ts, plt1[2], tq1, cmap=cmap, shading="nearest")  # nearest
    pcm2 = ax2.pcolormesh(ts, plt2[2], tq2, cmap=cmap, shading="nearest")  # nearest
    pcm3 = ax3.pcolormesh(ts, plt3[2], tq3, cmap=cmap, shading="nearest")  # nearest

    ax3.set_xlabel("time, yr")
    ax1.set_ylabel("latitude, degrees")
    ax2.set_ylabel("latitude, degrees")
    ax3.set_ylabel("latitude, degrees")
    ax1.set_yticks(np.linspace(-90, 90, 9, endpoint=True))
    ax2.set_yticks(np.linspace(-90, 90, 9, endpoint=True))
    ax3.set_yticks(np.linspace(-90, 90, 9, endpoint=True))
    plt.tight_layout()
    fig.colorbar(pcm1, ax=(ax1, ax2, ax3))

    plt.show()


def convergence_plot_single(
    tests,
    convtemps,
    val_name: str,
    val_range,
    val_unit: Optional[str] = None,
):
    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit
    # val_range = np.arange(val_min, val_max, val_step)
    # fit_range = np.linspace(min(val_range), max(val_range), 100, endpoint=True)
    # fit_vals = convtemps[0] * (1 - fit_range**2) ** (-1 / 4)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(val_range, tests)
    ax2.scatter(val_range, convtemps)
    # ax2.plot(fit_range, fit_vals, label="(1-e^2)^{-1/4} fit")
    ax2.set_xlabel(f"{val_name} {val_unit}")
    ax1.set_xticks(np.linspace(min(val_range), max(val_range), 11))
    ax2.set_xticks(np.linspace(min(val_range), max(val_range), 11))
    ax1.set_ylabel("Time to converge, years")
    ax2.set_ylabel("Global convergent temperature, K")
    plt.show()


def ecc_fit_plot(
    tests,
    convtemps,
    val_name: str,
    val_range,
    val_unit: Optional[str] = None,
):
    val_range = np.array(val_range)
    if val_unit is None:
        val_unit = ""
    else:
        val_unit = ", " + val_unit
    fit_range_1 = np.linspace(val_range[0], val_range[5], 100, endpoint=True)
    fit_vals_1 = convtemps[0] * (fit_range_1 / val_range[0]) ** (-1 / 2)

    fit_range_2 = np.linspace(val_range[6], val_range[-1], endpoint=True)
    fit_vals_2 = convtemps[6] * (fit_range_2 / val_range[6]) ** (-1 / 4)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1: plt.Axes
    ax2: plt.Axes
    ax1.scatter(val_range, convtemps, label="Model data")
    ax1.plot(fit_range_1, fit_vals_1, label="$a^{-1/2}$ fit on temperate")
    ax1.plot(fit_range_2, fit_vals_2, label="$a^{-1/4}$ fit on snowball")
    ax2.scatter(
        val_range[0:6],
        (convtemps[0:6] - convtemps[0] * (val_range[0:6] / val_range[0]) ** (-1 / 2))
        / convtemps[0:6],
        label="$a^{-1/2}$ residuals temperate",
    )
    ax2.scatter(
        val_range[6:],
        (convtemps[6:] - convtemps[6] * (val_range[6:] / val_range[6]) ** (-1 / 4))
        / convtemps[6:],
        label="$a^{-1/4}$ residuals snowball",
    )
    ax2.set_xlabel(f"{val_name} {val_unit}")
    ax1.set_xticks(val_range)
    ax2.set_xticks(val_range)
    ax2.set_ylabel("Residual")
    ax1.set_ylabel("Global convergent temperature, K")
    ax1.legend()
    # ax2.legend()
    plt.show()


def convergence_plot_dual(
    tests,
    convtemps,
    val_name_1,
    val_1_range,
    val_name_2,
    val_2_range,
    rtol=0.0001,
    val_unit_1=None,
    val_unit_2=None,
):
    if val_unit_1 is None:
        val_unit_1 = ""
    else:
        val_unit_1 = ", " + val_unit_1
    if val_unit_2 is None:
        val_unit_2 = ""
    else:
        val_unit_2 = ", " + val_unit_2

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


def orbital_animation():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from ClimateModel import orbital_model, line, r
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
    orbital_animation()
    # from filemanagement import load_config, read_file
    # from convergence import convergence_test

    # conf = load_config()
    # # q0 = read_file("single_omega/single_omega_2.41.npz")
    # # q1 = read_file("single_omega/single_omega_2.415.npz")
    # # q2 = read_file("single_omega/single_omega_2.42.npz")
    # # threecolourplot(q0, q1, q2, None, None, 1)

    # times, temps, degs = read_file("omega.npz")
    # dt = times[1] - times[0]
    # colourplot(degs, temps, times, None, None, 1)
    # plotdata(degs, temps, dt, 0, 365 * 1, 10)
    # print(convergence_test(temps, rtol=0.0001))
    # yearavgplot(degs, temps, dt, 90, 120, 1)

    # colourplot(degs, temps, times, None, None, 1, None, None)

    # one = read_files("testing_1.5.npz")
    # two = read_files("testing_3.npz")
    # three = read_files("testing_6.npz")

    # threecolourplot(one, two, three, 90, 120, 1)
