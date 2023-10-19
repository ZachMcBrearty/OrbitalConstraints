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


def colourplot(degs, temps, times, start=0, end=None, year_avg=1):
    if end is None:
        end = len(temps[0, :])

    fig, ax = plt.subplots(1, 1)
    cmap = "RdBu_r"

    ts = times[start : end : year_avg * 365]
    temp = temps[start:end, 1:]

    tq = np.average(temp.reshape((temp.shape[0], -1, 365 * year_avg)), axis=2)

    pcm = ax.pcolor(ts, degs, tq, cmap=cmap, shading="nearest")

    ax.set_xlabel("time, yr")
    ax.set_ylabel("latitude, degrees")
    ax.set_yticks(range(-90, 91, 15))
    fig.colorbar(pcm, ax=ax)

    plt.show()


if __name__ == "__main__":
    from filemanagement import load_config, read_files

    conf = load_config()
    times, temps, degs = read_files("testing.npz")
    dt = times[1] - times[0]

    # plotdata(degs, temps, dt, 0, 365 * 1, 10)
    yearavgplot(degs, temps, dt, 90, 100, 10)
    colourplot(degs, temps, times, 0, None, 5)
