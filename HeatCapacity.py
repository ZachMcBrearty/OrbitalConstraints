from typing import Optional

import numpy as np
import numpy.typing as npt

floatarr = npt.NDArray[np.float64]


def ThermalTimeScale(C, T, I):
    return C * T / I


def C(
    f_o: float | floatarr, f_i: float | floatarr, T: Optional[float | floatarr] = None
) -> float | floatarr:
    """Calculate heat capacity from ocean and ice fractions
    f_o: fraction of planet which is ocean
    f_i: fraction of ocean which is ice

    returns: heat capacity, J m^-2 K^-1
    """
    # C = ρ c_p Δl
    # ρ - density
    # c_p - heat capacity
    # Δl - depth of atmosphere
    C_l = 5.25 * 10**6  # J m^-2 K^-1 # heat capacity over land
    ## North et al 1983 ##
    # Δl = 75m
    C_o = 60 * C_l  # heat capacity over ocean
    C_i = 9.2 * C_l  # heat capacity over ice
    # ## WK97 ##
    # # 263 K < T < 273 K
    # if 263 < T < 273:
    #     C_i = 9.2 * C_l
    # elif T < 263:
    #     C_i = 2.0 * C_l
    # else:
    #     C_i = ???
    # # Δl = 50m
    # C_o = 40 * C_l
    return (1 - f_o) * C_l + f_o * ((1 - f_i) * C_o + f_i * C_i)


f_earth_10deg = np.array(
    [
        0.000,  # -90 - -80
        0.246,  # -80 - -70
        0.896,  # -70 - -60
        0.992,  # -60 - -50
        0.970,  # -50 - -40
        0.888,  # -40 - -30
        0.769,  # -30 - -20
        0.780,  # -20 - -10
        0.764,  # -10 -   0
        0.772,  #   0 -  10
        0.736,  #  10 -  20
        0.624,  #  20 -  30
        0.572,  #  30 -  40
        0.475,  #  40 -  50
        0.428,  #  50 -  60
        0.294,  #  60 -  70
        0.713,  #  70 -  80
        0.934,  #  80 -  90
    ]
)


def f_o(lat):
    return f_earth_10deg[np.int64(np.floor(17 * (lat / np.pi + 1 / 2)))]


def f_i(Temp):
    return 1 - np.exp((Temp - 273) / 10)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 30) * np.pi / 2
    temp: floatarr = np.exp(-5 * (x) ** 2) * 30 + 255
    F_o = f_o(x)
    F_i = f_i(temp)

    plt.plot(x, C(F_o, F_i, temp))

    plt.show()
