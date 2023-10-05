from typing import Optional

import numpy as np
import numpy.typing as npt

floatarr = npt.NDArray[np.float64]

stef_boltz = 5.670367 * 10**-8  # W m^-2 K^-4


def I_1(T, opt_thick: float = 1.0):
    return stef_boltz * T**4 / (1 + (3 * opt_thick / 4))


def opt_thick(T):
    return 0.79 * (T / 273) ** 3


def I_2(T, opt_thick_func):
    return stef_boltz * T**4 / (1 + (3 * opt_thick_func(T) / 4))


def I_3(T, A=2.033 * 10**5, B=2.094 * 10**3):
    """Linear emission profile
    A: erg cm^-2 s^-1
    B: erg cm^-2 s^-1 K^-1"""
    return A + B * T


def A_1(T):
    return 0.5 - 0.2 * np.tanh((T - 268) / 5)


def A_2(T):
    return 0.525 - 0.245 * np.tanh((T - 268) / 5)


def A_3(T):
    return 0.475 - 0.225 * np.tanh((T - 268) / 5)
