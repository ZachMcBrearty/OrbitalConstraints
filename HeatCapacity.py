from typing import Optional

import numpy as np
import numpy.typing as npt

floatarr = npt.NDArray[np.float64]


def ThermalTimeScale(
    Capacity: float | floatarr, Temp: float | floatarr, Ir_emission: float | floatarr
) -> float | floatarr:
    """Capacity: heat capacity of the system
    Temp: current temperature of the system
    Ir_emission: rate of energy loss from blackbody radiation

    returns: The characteristic time scale for thermal equilibrium"""
    return Capacity * Temp / Ir_emission


def get_C_func(spacedim):
    C_ref = 5.25 * 10**6
    C_l = C_ref
    # ## North et al 1983 ##
    # # Δl = 75m
    # C_o = 60 * C_l  # heat capacity over ocean
    # C_i = 9.2 * C_l  # heat capacity over ice
    C_i: floatarr = np.ones(spacedim) * 9.2 * C_ref
    # Δl = 50m -> 40 * C_l
    # Δl = 75m -> 60 * C_l
    C_o = 40 * C_ref

    def C(f_o: floatarr, f_i: floatarr, temp: floatarr) -> floatarr:
        """Calculate heat capacity from ocean and ice fractions
        f_o: fraction of planet which is ocean
        f_i: fraction of ocean which is ice

        returns: heat capacity, J m^-2 K^-1
        """
        # C = ρ c_p Δl
        # ρ - density
        # c_p - heat capacity
        # Δl - depth of atmosphere
        ## WK97 ##
        # 263 K < T < 273 K
        C_i[temp >= 263] = 9.2 * C_ref
        # T < 263 K
        C_i[temp < 263] = 2.0 * C_ref
        return (1 - f_o) * C_l + f_o * ((1 - f_i) * C_o + f_i * C_i)

    return C


f_earth_10deg = np.array(
    [
        0.000 + 0.05,  # -90 - -80, 0
        0.246,  # -80 - -70, 1
        0.896,  # -70 - -60, 2
        0.992,  # -60 - -50, 3
        0.970,  # -50 - -40, 4
        0.888,  # -40 - -30, 5
        0.769,  # -30 - -20, 6
        0.780,  # -20 - -10, 7
        0.764,  # -10 -   0, 8
        0.772,  #   0 -  10, 9
        0.736,  #  10 -  20, 10
        0.624,  #  20 -  30, 11
        0.572,  #  30 -  40, 12
        0.475,  #  40 -  50, 13
        0.428,  #  50 -  60, 14
        0.294,  #  60 -  70, 15
        0.713,  #  70 -  80, 16
        0.934,  #  80 -  90, 17
        0.934,  #  80 -  90, 18 # for use in f_o below
    ]
)


def f_o(lat: floatarr) -> floatarr:
    p = 17 * (lat / np.pi + 1 / 2)
    l = np.floor(p, dtype=int)
    s = p - l
    return f_earth_10deg[l] * (1 - s) + f_earth_10deg[l + 1] * s


# def f_o(lat):
#     return f_earth_10deg[np.int64(np.floor(18 * (lat / np.pi + 1 / 2)))]


def f_i(Temp: floatarr) -> floatarr:
    q = 1.0 - np.exp((Temp - 273) / 10, dtype=float)
    q[q < 0] = 0.0
    q[q > 1] = 1.0
    return q


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # x = np.linspace(-1, 1, 30) * np.pi / 2
    # x = np.linspace(-1, 1, 200)
    lats = np.linspace(-1, 1, 200) * np.pi / 2
    # temp: floatarr = np.ones_like(lats) * 350  #
    temp: floatarr = np.exp(-10 * (lats) ** 2, dtype=float) * 50.0 + 250.0
    # F_o = f_o(lats)
    F_o = np.ones_like(lats) * 0.7

    F_i = f_i(temp)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    C = get_C_func(len(lats))
    ax1.plot(lats, 1 / C(F_o, F_i, temp))
    ax1.set_ylabel(r"1/Heat capacity, J m$^{-2}$ K$^{-1}$")
    ax2.plot(lats, temp)
    ax2.axhline(273, ls="--", label=r"0$^{\circ}$C")
    ax2.set_ylabel(r"Temperature, K")
    ax3.plot(lats, F_o, label=r"Ocean fraction, $f_o$")
    ax3.plot(lats, F_o * F_i, label=r"Ice fraction, $f_o f_i$")
    ax3.set_ylabel(r"Fraction, unitless")
    ax3.set_xlabel(r"Latitude, radians")

    ax2.legend()
    ax3.legend()
    fig.text(
        0.15,
        0.95,
        r"Heat capacity of Earth from a Gaussian temperature and Ocean fraction",
    )

    plt.show()
