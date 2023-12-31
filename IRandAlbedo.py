from typing import Optional

import numpy as np
import numpy.typing as npt

from Constants import STEPH_BOLTZ

floatarr = npt.NDArray[np.float64]


def I_1(T: float | floatarr, opt_thick: float = 1.0) -> float | floatarr:
    """Simplest model, black body radiation with constant reflection / reradiation from atmosphere
    T: temperature, K
    opt_thick: the optical thickness of the atmosphere, unitless

    return: simple IR emission, W m^-2"""

    return STEPH_BOLTZ * T**4 / (1 + (3 * opt_thick / 4))


def opt_thick(T: float | floatarr) -> float | floatarr:
    """Temperature variable reflection / reradiation from atmosphere
    T: Temperature, K

    return: variable optical thickness due to temperature"""
    if isinstance(T, np.ndarray):
        o = 0.79 * (T / 273) ** 3
        o[o > 1] = 1
        o[o < 0] = 0
        return o
    else:
        if (q := 0.79 * (T / 273) ** 3) > 1:
            return 1
        elif q < 0:
            return 0
        else:
            return q


def I_2(T):
    """Blackbody radiation with temperature variable reflection / reradiation
    T: temperature, K

    return: variable IR emission with temperature, W m^-2"""
    return STEPH_BOLTZ * T**4 / (1 + (3 * opt_thick(T) / 4))


def I_3(T: float | floatarr, A=2.033 * 10**5, B=2.094 * 10**3) -> float | floatarr:
    """Linear emission profile
    A: erg cm^-2 s^-1
    B: erg cm^-2 s^-1 K^-1

    return: A + B * T, erg cm^-2 s^-1"""
    return A + B * T


def A_1(T: float | floatarr) -> float | floatarr:
    """All 3 albedo functions are simple tanh functions interpolating between different values
    A_1 -> 0.3 - 0.7
    T: temperature

    returns: albedo based loosely on whether the land is covered by ice or not"""
    return 0.5 - 0.2 * np.tanh((T - 268) / 5)


def A_2(T: float | floatarr) -> float | floatarr:
    """All 3 albedo functions are simple tanh functions interpolating between different values
    A_2 -> 0.28 - 0.77
    T: temperature

    returns: albedo based loosely on whether the land is covered by ice or not"""
    return 0.525 - 0.245 * np.tanh((T - 268) / 5)


def A_3(T: float | floatarr) -> float | floatarr:
    """All 3 albedo functions are simple tanh functions interpolating between different values
    A_3 -> 0.25 - 0.7
    T: temperature

    returns: albedo based loosely on whether the land is covered by ice or not"""
    return 0.475 - 0.225 * np.tanh((T - 268) / 5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 30) * np.pi / 2
    temp: floatarr = np.exp(-5 * (x) ** 2) * 30 + 255

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(x, I_1(temp), label="I_1")
    ax1.plot(x, I_2(temp), label="I_2")
    ax1.set_ylabel(r"IR Emission, J m$^{-2}$ s$^{-1}$")
    ax1.legend()

    ax2.plot(x, A_1(temp), label="A_1")
    ax2.plot(x, A_2(temp), label="A_2")
    ax2.plot(x, A_3(temp), label="A_3")
    ax2.set_ylabel(r"Albedo, unitless")
    ax2.legend()

    ax3.plot(x, temp)
    ax3.axhline(273, ls="--", label=r"0$^{\circ}$C")
    ax3.set_ylabel(r"Temperature, K")
    ax3.set_xlabel(r"Latitude, radians")
    ax3.legend()

    fig.text(
        0.15,
        0.95,
        r"IR emission and Albedo for a gaussian temperature distribution",
    )

    plt.show()
