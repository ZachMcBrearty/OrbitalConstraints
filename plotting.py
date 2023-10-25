import matplotlib.pyplot as plt
import numpy as np

yeartosecond = 365.25 * 24 * 3600  # s / yr


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
    a[0].set_marker("x")
    ax.plot(
        degs,
        302.3 - 45.3 * np.sin(np.deg2rad(degs)) ** 2,
        marker=".",
        ls="--",
        label="fit",
    )
    ax2.plot(degs, np.abs(yr_avg - (302.3 - 45.3 * np.sin(np.deg2rad(degs)) ** 2)))
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
    ax1.set_yticks(np.linspace(-90, 90, 13, endpoint=True))
    ax2.set_yticks(np.linspace(-90, 90, 13, endpoint=True))
    ax3.set_yticks(np.linspace(-90, 90, 13, endpoint=True))
    plt.tight_layout()
    fig.colorbar(pcm1, ax=(ax1, ax2, ax3))

    plt.show()


if __name__ == "__main__":
    from filemanagement import load_config, read_files
    from convergence import convergence_test

    conf = load_config()
    times, temps, degs = read_files("testing_1.5_low.npz")
    # dt = times[1] - times[0]

    # plotdata(degs, temps, dt, 0, 365 * 1, 10)
    # print(convergence_test(temps, rtol=0.0001))
    # yearavgplot(degs, temps, dt, 90, 120, 1)
    colourplot(degs, temps, times, 95, 150, 1, None, None)
    # colourplot(degs, temps, times, None, None, 1, None, None)

    # one = read_files("testing_1.5.npz")
    # two = read_files("testing_3.npz")
    # three = read_files("testing_6.npz")

    # threecolourplot(one, two, three, 90, 120, 1)
